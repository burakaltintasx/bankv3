import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer

#türkçe ses eğitim
tokenizer = Wav2Vec2Tokenizer.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')
model = Wav2Vec2ForCTC.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')

audio_value = st.audio_input("Record a voice message")

if audio_value:
    #st.audio(audio_value)

    x, sr = librosa.load(audio_value, sr=16000)

    input_values = tokenizer(x, return_tensors="pt").input_values
    logits = model(input_values).logits
    pretrained = torch.argmax(logits, dim=-1)
    sonuc = tokenizer.decode(pretrained[0])
    st.write(sonuc)

mesaj=st.text_area("Yapmak istediğiniz işlemi kısaca açıklayın")
btn=st.button("Getir")
if btn:
    df=pd.read_csv('bankv3.csv')
    df=df[['sorgu','label']]
    cv=CountVectorizer(max_features=250)
    rf=RandomForestClassifier()
    x=cv.fit_transform(df['sorgu']).toarray()
    y=df['label']
    model=rf.fit(x,y)
    mesajvektor=cv.transform([mesaj]).toarray()
    sonuc=model.predict(mesajvektor)
    st.write(sonuc[0])