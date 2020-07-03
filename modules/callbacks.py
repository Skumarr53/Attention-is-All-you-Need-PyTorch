from statistics import mean
from fastai.callback import Callback
from fastai.vision import *
from pathlib import  Path, posixpath
from pdb import set_trace
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from torch.autograd import Variable
from pdb import  set_trace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmoothLabelCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    2. Calculate loss
    """

    def __init__(self, label_smoothing=0.0):
        super(SmoothLabelCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        # When label smoothing is turned on, KL-divergence is minimized
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            self.criterion = nn.NLLLoss()
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        # Map the output to (0, 1)
        dec_outs = dec_outs[0]
        scores = self.LogSoftmax(dec_outs)
        # n_class
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = self.criterion(scores.view(-1, scores.size(-1)), gtruth)
        return loss


class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx = 1, smoothing = 0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False) #size_average changed to reduction
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = vocab_size
        self.true_dist = None
        
    def forward(self, input,targets):
        pred,_,_,_,decode_lengths = input
        x,y = pred.contiguous().view(-1, pred.size(-1)), targets.contiguous().view(-1)
        assert self.size == x.size(1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size - 0))
        true_dist.scatter_(1, y.data.unsqueeze(1), self.confidence)
        #true_dist[:,self.padding_idx] = 0
        mask = torch.nonzero(y.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(F.log_softmax(x, dim=-1), Variable(true_dist, requires_grad=False))/decode_lengths.sum().item()


def CrossEntropy_loss(input,targets):
    pred,_,_,_,decode_lengths = input
    x,y = pred.contiguous().view(-1, pred.size(-1)), targets.contiguous().view(-1)
    loss = nn.CrossEntropyLoss().to(device)(F.log_softmax(x, dim=-1), y)
    return  loss #loss(pred.data.long(), targets.data.long())

def seq2seq_acc(input, targ, pad_idx=1):
    out = input[0]
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    out = out.argmax(2)
    return (out==targ).float().mean()


class TeacherForcing(LearnerCallback):
    
    def __init__(self, learn, end_epoch):
        super().__init__(learn)
        self.end_epoch = end_epoch
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train: return {'last_input': [last_input, last_target]}
    
    def on_epoch_begin(self, epoch, **kwargs):
        self.learn.model.pr_force = 1 - epoch/self.end_epoch



# Bleu metirc

class NGram():
    def __init__(self, ngram = 4, max_n=5000): self.ngram,self.max_n = ngram,max_n
    def __eq__(self, other):
        if len(self.ngram) != len(other.ngram): return False
        return np.all(np.array(self.ngram) == np.array(other.ngram))
    def __hash__(self): return int(sum([o * self.max_n**i for i,o in enumerate(self.ngram)]))

def get_grams(x, n, max_n=5000):
    return x if n==1 else [NGram(x[i:i+n], max_n=max_n) for i in range(len(x)-n+1)]


def get_correct_ngrams(pred, targ, n, max_n=5000):
    pred_grams,targ_grams = get_grams(pred, n, max_n=max_n),get_grams(targ, n, max_n=max_n)
    pred_cnt,targ_cnt = Counter(pred_grams),Counter(targ_grams)
    return sum([min(c, targ_cnt[g]) for g,c in pred_cnt.items()]),len(pred_grams)


class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = 'bleu'
    
    def on_epoch_begin(self, **kwargs):
        self.pred_len,self.targ_len,self.corrects,self.counts = 0,0,[0]*4,[0]*4
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        out = last_output[0].argmax(dim=-1)
        for pred,targ in zip(out.cpu().numpy(),last_target.cpu().numpy()):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c,t = get_correct_ngrams(pred, targ, i+1, max_n=self.vocab_sz)
                self.corrects[i] += c
                self.counts[i]   += t
    
    def on_epoch_end(self, last_metrics, **kwargs):
        precs = [c/t for c,t in zip(self.corrects,self.counts)]
        len_penalty = exp(1 - self.targ_len/self.pred_len) if self.pred_len < self.targ_len else 1
        bleu = len_penalty * ((precs[0]*precs[1]*precs[2]*precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)


class BleuMetric(Callback):

    def on_epoch_begin(self, **kwargs):
        self.bleureferences = list()
        self.bleucandidates = list()

        
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output[0].argmax(dim=-1)
        decode_lengths = last_output[4].tolist()
        preds, decode_lengths = list(preds), decode_lengths
        hypotheses = list()
        references = list()
        for i,cap in enumerate(preds): hypotheses.append([str(x) for x in cap.tolist()[:decode_lengths[i]]])
        for i,cap in enumerate(last_target): references.append([str(x) for x in cap.tolist()[:decode_lengths[i]]])
        #for i,cap in enumerate(pred_words): hypotheses.append([x for x in cap.tolist() if x not in {self.vocab['xxunk'], self.vocab['xxpad'], self.vocab['xxbos'], self.vocab['xxeos'],self.vocab['xxfld'],self.vocab['xxmaj'],self.vocab['xxup'],self.vocab['xxrep'],self.vocab['xxwrep']}])
        self.bleureferences.extend(references)
        self.bleucandidates.extend(hypotheses)

        

        
    def on_epoch_end(self, last_metrics, **kwargs):
        assert len(self.bleureferences) == len(self.bleucandidates)
        bleu4 = corpus_bleu(self.bleureferences, self.bleucandidates)
        return add_metrics(last_metrics,bleu4)