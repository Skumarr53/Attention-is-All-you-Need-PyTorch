from modules.model import *
from fastai import *
import streamlit as st
from  pathlib import Path
import logging
import urllib
#import settings

#from settings import * # import

model_path = Path('./models')
learn = load_learner(model_path, 'Fastai-resnet-34.pkl')
defaults.device = torch.device('cpu')

def translate(text):
    enc_inp = torch.tensor(databunch.x.process_one(text)).unsqueeze(0)
    dec_out = [2]
    dec_len = torch.tensor([50])

    while len(dec_out) < 50:
        set_trace()
        dec_input = torch.as_tensor([dec_out], dtype=torch.long)
        enc_inp,dec_input,dec_len=[i.to('cuda') for i in (enc_inp,dec_input,dec_len)]
        with torch.no_grad():
            out = learn.model(enc_inp[:,1:],dec_input,dec_len)
        predict = out[0].squeeze(0).argmax(1)[-1].detach().cpu().numpy()
        dec_out.append(int(predict))

    dec_out = torch.as_tensor([dec_out], dtype=torch.long).unsqueeze(0)
    out_text = learn.data.train_ds.y.reconstruct(dec_out)
    return out_text



def main():
    st.title("French to English Translator")
    st.subheader("translate freanch queries to english")

    # get input from user
    query = st.text_area("Enter Your French Query","Type Here")

     
    if st.button("Translate"):
        if query == "":
            st.success("No text found")
        else:
            st.success(translate(query));


if __name__=='__main__':
    main()

'''
dec_inp = start_tok[shape=(1,1)]
enc_inp = inp_sent[shape=(1,len(sent))]

'''