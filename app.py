import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import seaborn as sns


df = pickle.load(open('styles.dat','rb')) 
dftsne = pickle.load(open('tsnedf.dat','rb'))
dftop = pickle.load(open('top50for44k.dat','rb'))
datadir = 'train/'
#image = Image.open(datadir+'10000.jpg')

#st.title('Recommendation')


def recommendation():
    st.title('Recommendation')
    x = df.sample()
    cap = x['image'].tolist()[0]
    image = Image.open(datadir+cap)
    st.image(image, caption=cap, width = 200)
    row = dftop.loc[x.index[0]]
    for i in range(0,2):
        cols = st.beta_columns(3)
        cols[0].image(Image.open(datadir+df.loc[row[i*3][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)
        cols[1].image(Image.open(datadir+df.loc[row[i*3+1][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)
        cols[2].image(Image.open(datadir+df.loc[row[i*3+2][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)

# def recommendation2(rowid):
#     x = df.loc[rowid]
#     cap = x['image']#.tolist()[0]
#     image = Image.open(datadir+cap)
#     st.image(image, caption=cap, width = 200)
#     row = dftop.loc[rowid]
#     for i in range(0,2):
#         cols = st.beta_columns(3)
#         cols[0].image(Image.open(datadir+df.loc[row[i*3][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)
#         cols[1].image(Image.open(datadir+df.loc[row[i*3+1][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)
#         cols[2].image(Image.open(datadir+df.loc[row[i*3+2][0]].image),caption='similarity: '+str(row[i*3][1]), use_column_width=True)


#@st.cache#(suppress_st_warning=True)
def embedding_plot(Category):
    st.title('TSNE Plot')
    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                    hue=Category,
                    data=dftsne,
                    legend="full",
                    alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig)


x=0
st.sidebar.write('Click to See Recommendation')
# rowid = st.sidebar.number_input('Enter Row Number',
#                         min_value=0,
#                         max_value=44441,
#                         step=1)
# recommendation2(rowid)
# st.sidebar.write('Or')
if st.sidebar.button("Random Image"):
    recommendation()



st.sidebar.text("")

st.sidebar.write('Image Embedding Plot Options')
options = st.sidebar.radio(
     'Hue Category',
    ['masterCategory', 'subCategory', 'articleType', 'baseColour','season','usage'],
    index=0)


embedding_plot(options)
#st.pyplot()

st.sidebar.write('Sizing')
max_width = st.sidebar.slider('Column width',100, 2000, 1200, 100)
    
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        
    }}

</style>
""",
        unsafe_allow_html=True,
    )
st.set_option('deprecation.showPyplotGlobalUse', False)
