# Attention is all you need

![](Snapshots/the-annotated-transformer.png)

## Transformers?, why not RNNS ?
### RNN based translation

A basic Seq2seq model consits of an encoder and decoder. Model takes input sentence into encoder and encodes word by word in each step and outputs hidden state (eh[t]) that stores the context of sentence till that point. so final hiden state at the end (eh[T]) stores context of entire sentence.

This hidden state (eh[T]) becomes input for decoder that decodes the context vector(eh[T]) and produces translated sentence word by word. At each step, decoder outputs word and hidden state(dh[t]) which is will used for generating next word.

![](Snapshots/RNN_based_encoderDecoder.png)
 
### RNN based translation with Atention
In case of above model with Attention it differs in following things:

Instead of last hidden state, all the states (eh[0], eh[1]...,eh[T]) at every step along with final context vector (eh[T]) are passed into decoder. Idea here is each hidden state is mostly associated with certain word in the input sentence. using all the hidden state gives better translation.

At every time step in Decoding phase, Socres are computed for every hidden states (eh[t]) that stor how relevant is particular hidden state in predicting word at current step(t) in this way more importance is given to hidden state that is relevant in predicting current word.

ex: when decoder predicts ist word more importance should be given to 1st hidden, 2nd hidden state (depends on the language struture) od all input hidden states.

![](Snapshots/RNN_with_attention.png)

This method is significant improvement over prevoius but lacks parallelization capability and thus greater time complexity.

## Transform Theory

It is used for translation i.e seq2seq as Attention model. It is different in following aspects.

1. It can be parallelised unlike Attention. Attention uses RNN as base for encoding the info i.e hidden state. So, last word has to wait till all previous words are processed or encoded in context vector.

2. In transofrmer there is not such concept as hidden state. trnasformer 'Self attention' captures association of certain word with other words in the input sentences. This is achieved by three vectors termed as query, key
, value obtained by multiplying **word embedding** with the weight matrices [unknown param to be estimated] one for each query(q), key(k) and value(v).

Note: To embed word relative position in the sentence. word embedding is added with postional encoding that preserves word position related information. 

3. Then using the q, k and v, attention scores are computed for each word with other words. So each word has one attention vector i.e each words have their own q, k and v matrices.

![](Snapshots/Attention_compute.png)

Wq, Wk, Wv are parameter matrices to be learned to get q, k and v upon multiplying with input embeddings

Attention scores = softmax( dot_product( q[j],k[i] ) / sqrt(T) ) * v(j)

T - dimesion of the key

![](Snapshots/Attention_putput.png)

We can see how attention screens out non relevant inputs for generating each words.

Since inputs for predicting each word in independent of each other they can be parallelized.

Till now we have seen single head attention. we can also use multiple sets of q, k, v for each word for computing individual Attention scores and Finally resulting matrices are concatenated This is called Multi head attention. 

![](Snapshots/MultiHead_Attention.png)

* Attention Scores (vectors) are feed into feed forward network with weight matrices (Wo) to make attention output same dimension as the word embedding. 

As we pass input embedding through many layers postional infomation may be decay. To make up for this we add attention output with input rataining position related information.

This is Encoder's final output is demension as input encoder. We can feed output to another encoder to make model robust. encoder can be stacked in series. Final, output from the last encoder becomes input for decoder. 



simlarly multiple decoders can be stacked in series where first decoder users final output of encoder. Last decoder outputs prediction word.

![](Snapshots/Decoder_stru.png)

Inputs to the decoder is passed through the **Masked Multi head Attention** that prevent future words to be part of the attention.

Fro decoder multihead attention layer query matrices are generated within from (prevoius decoder layer) but key and value come from the encoder. Since query changes at every step, so does Attention  telling which ![](Snapshots/words to focus on input words to generate current word.

![](Snapshots/Decode_layer.png)

outputs are passed fully comnnected layers same as encoder. Then Linear and softmax to expand dimesion to vocabulary size (Linear) and corvert values to probabilities (softmax). word corresponding index of max probabilty will be output.

![](Snapshots/Final_DecoderPredOutput.png)



### Intiuation behind using various components:

**Positional Encoding**: As the **Tranformer** architecture does not have component to represent sequential nature of the data. This layer is used to inject relative positions of the word in the sequence.

**Self-attention**: word embedding is broke down into *query*, *key* and *value* of same dimension. During training phase, these matrices learns how much other words value it. *query* queries other words and get feedback as *key* which then dot producted with *value* to get the score. This is performed against all other words 

**Multi-Head Attention**: Multiple heads provided a greater fexibility in understanding context as each one of it looks at same input embedding in different context (different Q, K, V) and brougut to same size as input embedding by concatinating. 


**Masked Self-Attention**: Encoder component needs to look at each word in the input sentence to understand context but while decoding, let say predicting 'i' th,  it should be only be allowed to look at previous i-1 words ,otherwise, it would be cheating. It done by masking.

**Encoder-Decoder Attention**: Decoder has to relay on Encoder input for understanding context of the complete sentence. Its achieved by allowing decoder query the encoded embeddings (*key* and *value*) that stores both positional and contextual information. Deocder predicts based on this embedding.

**Encoder-Decoder Stack**: One thing to notice in the **Transformer** network is the dimesions of the encoded embeddings (output of encoder) remains same. In other words it can said that Encoded embedding in as improvised representation of orinal data. It can still be improved by stacking similar layer in sequence.  


``` python
def make_model(src_vocab, tar_vocab, N = 6, d_model = 512,
               d_ff = 2048, h = 8, dropout = 0.1):
    
    c  = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionalEncoding(d_model, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout),N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), 
                             dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tar_vocab), c(position)),
        Generator(d_model, tar_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
    
```

smaple data:

 | fr | en
-|----|---
 | Quelle sorte de poste aimeriez-vous occuper? | What sort of job would you like to be in?
 | What do adult attachment scales measure? | What do adult attachment scales measure?
 | Quel mécanisme utiliser pour mettre en œuvre cette protection (par exemple, les principes UDRP ou un autre mécanisme)? | What mechanism should be used to implement protection (for example, the UDRP or some other mechanism)?
 | Qu'arrive-t-il si les registres d'un organisme de bienfaisance ne sont pas satisfaisants? | What happens if a charity's records are inadequate?
 | À votre avis, pourquoi est-il important que les gouvernements gèrent les terres et les rivières? | What type of activities threaten the animals or plants in a national park?


input Parametes:

``` python
{'src_vocab_size': 22188,
 'tgt_vocab_size': 14236,
 'd_model': 300,
 'd_ff': 2048,
 'd_k': 64,
 'd_v': 64,
 'n_heads': 6,
 'n_layers': 6}
```

1. **src_vocab_size** - source language vocabulary size
2. **tgt_vocab_size** - target language vocabulary size
3. **N** - No of layers Multi-Head attention units inside Encoder and decoder
4. **d_model** - dimension of hidden units
5. **d_ff** - dimension of the decoder's feed-forward output
6. **d_k** - dimension of *key* vector 
7. **d_v** - dimension of *value* vector
8. **n_heads** - No of units in the Multihead attention
9. **n_layers** - No of layers Multi-Head attention units inside Encoder and decoder

### Model Architecture:

1. **copy.deepcopy** is function that creates copy of objects such that target if independent of original i.e changes made in the target do not reflect in the original.

2. **MultiHeadedAttention**<br>
* Class takes key, query and value (optional mask in case of decoder) vectors. 
* **LayerNorm** - normalization 

``` python    
**input - outputs**
    src = torch.Size([64, 30])
    tar = torch.Size([64, 30])
    src_mask = torch.Size([64, 1, 30])
    tar_mask = torch.Size([64, 1, 30])
    embedding = torch.Size([22188, 512])
    embedding_out = torch.Size([64, 30, 512])
    PositionalEncoding = torch.Size([64, 30, 512])
    n_EncoderLayer = 6
    EncoderLayer = torch.Size([64, 30, 512]) - 
    MultiHeadedAttention: 
        query, key, value = torch.Size([64, 30, 512])
        tranformed: query, key, value = torch.Size([64, 8, 30, 64])
        n_heads

        attetion scores = torch.Size([64, 8, 30, 64])
        concanting_scores = torch.Size([64, 30, 512])

        post attention feed_froward (attntion dim to input embed dim) = torch.Size([64, 30, 512])
        layer_norm + residual_conn + dropout layer = torch.Size([64, 30, 512])

        positional_encoding(pe) =  torch.Size(['max_seq_len',512])
        decoder_out = torch.Size([64, 28, 512])
        prob_mat = torch.Size([64, 28, 14236])
```

## Training

epoch | train_loss | valid_loss | seq2seq_acc | bleu | time
------|------------|------------|-------------|------|-----
0 | 2.548614 | 2.653859 | 0.609423 | 0.433331 | 01:28
1 | 2.037472 | 2.200202 | 0.656494 | 0.467307 | 01:29
2 | 1.715760 | 1.906313 | 0.683405 | 0.490627 | 01:30
3 | 1.332565 | 1.701444 | 0.707107 | 0.514892 | 01:34
4 | 1.062245 | 1.594243 | 0.722780 | 0.536479 | 01:36
5 | 0.713107 | 1.555298 | 0.735008 | 0.557943 | 01:34
6 | 0.394070 | 1.591297 | 0.743378 | 0.576106 | 01:31
7 | 0.177876 | 1.653029 | 0.746438 | 0.584508 | 01:32
8 | 0.075402 | 1.695086 | 0.748850 | 0.589291 | 01:31
9 | 0.042065 | 1.710636 | 0.749605 | 0.590379 | 01:32


### References:
1. [Attention Is All You Need - Paper (arxiv)](https://arxiv.org/abs/1706.03762)

2. [**harvardnlp** - The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
2. [Fastai - Introduction to Transformer](https://www.youtube.com/watch?v=KzfyftiH7R8&list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9&index=18)

3. [Transformer (Attention is all you need) - Minsuk Heo 허민석](https://www.youtube.com/watch?v=z1xs9jdZnuY&t=182s)

    def forward(self, dec_outs, labels):
        # Map the output to (0, 1)
        dec_outs = dec_outs[0]
        set_trace()
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
        loss = self.criterion(scores, gtruth)
        return loss