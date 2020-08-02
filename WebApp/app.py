from modules.model import *
from modules.callbacks import *
from fastai.text import *
from fastai import *
from fastai.vision import *
import torch
import streamlit as st
from  pathlib import Path
import logging
import urllib
from pdb import set_trace
#import settings

#fastai.defaults.device = torch.device('cpu')
#from settings import * # import
defaults.device = torch.device('cpu')
model_path = Path('./models')
learn = load_learner(model_path, 'export.pkl')
learn.model = learn.model.to('cpu')


def translate(text):
    enc_inp = torch.tensor(learn.data.x.process_one(text)).unsqueeze(0)
    dec_out = [2]
    dec_len = torch.tensor([50])

    while len(dec_out) < 50:
        dec_input = torch.as_tensor([dec_out], dtype=torch.long)
        enc_inp,dec_input,dec_len=[i.to('cpu') for i in (enc_inp,dec_input,dec_len)]
        with torch.no_grad():
            out = learn.model(enc_inp[:,1:],dec_input,dec_len)
        predict = out[0].squeeze(0).argmax(1)[-1].detach().cpu().numpy()
        dec_out.append(int(predict))

    
    dec_out = [x for x in dec_out if x not in [1,2]]
    out_text = learn.data.train_ds.y.reconstruct(torch.tensor(dec_out))
    return str(out_text)



def main():
    st.title("French to English Translator")

    # get input from user
    query = st.text_area("Enter Your French Query","Type Here")
     
    if st.button("Translate"):
        if query == "":
            st.success("No text found")
        else:
            st.success(translate(query));
            

if __name__=='__main__':
    main()