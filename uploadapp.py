import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import pairwise_distances
import cv2
from tensorflow.keras.preprocessing.image import load_img

df = pickle.load(open('styles.dat','rb')) 

embeddings = pickle.load(open('resnetembreindexed.dat','rb'))

resnet = tf.keras.models.load_model('resnet.h5')
img_width, img_height, _ = 300,400, 3 


def img_path(img):
    return 'train/'+img

def load_image(img, resized_fac = 0.1):
    img     = cv2.imread(img_path(img))
    return img

def get_embedding(model, img_name):
    # Reshape
    img = load_img(img_name, target_size=(img_width, img_height))
    # img to Array
    x   = np.array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    #print(x.shape)
    return model.predict(x).reshape(-1)



datadir= 'train/'

def recommendation(filen):
    st.title('Recommendation')
    imemb = get_embedding(resnet, filen)
    cosine_sim = 1-pairwise_distances(embeddings, [imemb], metric='cosine')
    
    sort_ind = np.argsort(-cosine_sim,axis=0)
    
    top6 = sort_ind[1:7]
    #st.write(datadir+df.loc[top6[1]].image)
    for i in range(0,2):
        cols = st.beta_columns(3)
        #st.write(datadir+df.loc[int(top6[i*3])].image)
        cols[0].image(Image.open(datadir+df.loc[int(top6[i*3])].image),caption='similarity: '+str(float(cosine_sim[int(top6[i*3])])), use_column_width=True)
        cols[1].image(Image.open(datadir+df.loc[int(top6[i*3+1])].image),caption='similarity: '+str(float(cosine_sim[int(top6[i*3+1])])), use_column_width=True)
        cols[2].image(Image.open(datadir+df.loc[int(top6[i*3+2])].image),caption='similarity: '+str(float(cosine_sim[int(top6[i*3+2])])), use_column_width=True)

st.title("Uploaded Image")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    imag = Image.open(uploaded_file)
    st.image(imag, width=250)
    recommendation(uploaded_file)



