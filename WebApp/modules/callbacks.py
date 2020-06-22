from statistics import mean
from transformer.model import Translation
from fastai.callback import Callback
from fastai.vision import *
from pathlib import  Path, posixpath
from fastai.text import *
from pdb import set_trace
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable

defaults.device = torch.device('cpu')
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seq2seq_collate(samples, pad_idx=1, pad_first=True, backwards=False):
    
    # unpack samples to tuples
    samples = to_data(samples)
    
    # find max len of x, y batch wihich decides inp seq length
    max_len_x,max_len_y = max([len(s[0]) for s in samples]),max([len(s[1]) for s in samples])
    
    max_len = max(max_len_x,max_len_y)
    
    # create a dummy tensor of height batch_size and width max_len with padded value
    res_x = torch.zeros(len(samples), max_len).long() + pad_idx
    res_y = torch.zeros(len(samples), max_len).long() + pad_idx
    
    # if backwards activate reverse mode used in bi-directional   
    if backwards: pad_first = not pad_first
    
    # fill vocabulary indices
    for i,s in enumerate(samples):
        if pad_first: 
            res_x[i,-len(s[0]):],res_y[i,-len(s[1]):] = LongTensor(s[0]),LongTensor(s[1])
        else:         
            res_x[i, :len(s[0])],res_y[i, :len(s[1])] = LongTensor(s[0]),LongTensor(s[1])
    
    # flip backward if backwards = True
    if backwards: res_x,res_y = res_x.flip(1),res_y.flip(1)
        
    res_x_mask = (res_x != pad_idx).unsqueeze(-2)
    res_y_mask = None
    
    # target mask creation
    if res_y is not None:
        dec_y = res_y[:, :-1]
        tar_y = res_y[:, 1:]
        decode_lengths = torch.tensor([len(s[1]) for s in samples])-1
    return (res_x[:,1:],dec_y,decode_lengths), tar_y

class Seq2SeqDataBunch(TextDataBunch):
    " decorator adds following method additionally to base class 'TextDataBunch"
    
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=8, val_bs:int=None, pad_idx=1,
               dl_tfms=None, pad_first=False, device:torch.device=device, no_check=True, backwards:bool=False, **dl_kwargs)-> DataBunch:
        
        "Function takes pytorch dataset object transforms into 'databunch' for classification and cls will allow to access parent class methods just  like 'self'"
        
        # store dataset obj into list
        datasets = cls._init_ds(train_ds,valid_ds,test_ds)
        val_bs = ifnone(val_bs, bs) #returns val_bs if none bs
        
        # stores raw seq of unequal len into tensor with padding
        # below function takes batches output from Dataloader returns padded tensor
        collate_fn = partial(seq2seq_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        
        #sampler function: generater takes dataset then sorts and returns sorted indices    
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        
        # train data loader obj with Sortish sampler(sort with some randomness)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        
        # other dataloaders with pure sorting append into dataloaders list
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key = lengths.__getitem__)
            dataloaders.append(DataLoader(ds,batch_size=val_bs,
                                          sampler = sampler,**dl_kwargs))
        
        return cls(*dataloaders, path = path, device = device, collate_fn=collate_fn, no_check = no_check)

class Seq2SeqTextList(TextList):
    _bunch = Seq2SeqDataBunch
    _label_cls = TextList

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

def LabelSmoothingCrossEntropy_func(input,targets):
    x,y = input,targets
    loss = SmoothLabelCritierion(label_smoothing=0.1).to(device)(x,y)
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
    pred = pred.view(-1, pred.size(-1))
    targs = targets.contiguous().view(-1)
    loss = nn.CrossEntropyLoss().to(device)(pred, targs)
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