from pathlib2 import Path
import torch
import torch.nn as nn
import fasttext as ft
from tqdm.notebook import tqdm
import numpy as np
import pickle
import collections

path = Path()
TMP_PATH = path/'fr-2-en-data/tmp/'

def load_ids(pre):
    ids = np.load(TMP_PATH/f'{pre}_ids.npy',allow_pickle=True)
    itos = pickle.load(open(TMP_PATH/f'{pre}_itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    return ids,itos,stoi

lang_models = ['en','fr']

def create_emb(vecs, itos, em_sz=300, mult=1.):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    vec_dic = {w:vecs.get_word_vector(w) for w in vecs.get_words()}
    miss = []
    for i,w in tqdm(enumerate(itos)):
        try: wgts[i] = tensor(vec_dic[w])
        except: miss.append(w)
    return emb

for lang in lang_models:
    _,itos,_ = load_ids(lang)
    vecs = ft.load_model(str(f'/home/skumar/.nlp_wordembeddings/cc.{lang}.300.bin'))
    emb = create_emb(vecs,itos)
    torch.save(emb,f'models/{lang}_emb.pth')
    del vecs