import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

covid = st.beta_container()
model = st.beta_container()
visual = st.beta_container()
graph = st.beta_container()
safety = st.beta_container()
about = st.beta_container()



with covid:
    st.title('XCEPTION CNN BASED COVID DETECTION SYSTEM USING CHEST X-RAY IMAGES')
    
    st.markdown('Covid pandemic has brought loss in both health and wealth to the world')
    cov_data = pd.read_csv('covid_19_data.csv')
    st.write(cov_data.head())

    confirmed_dist = pd.DataFrame(cov_data['New cases'].value_counts()).head(50)
    st.bar_chart(confirmed_dist)
       
with model:
	st.header('COVID-19 DETECTION MODEL')
	st.text('Press the Predict  button to load the model and load images from directory Test')
	Category =['COVID','NORMAL']
	model =  tf.keras.models.load_model('model.h5')
	model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

	if st.button('predict'):
            Test_path = 'Test'

            Category = ['COVID','NORMAL']

            input_path = 'Coviddataset/Train'

            Target_path = 'Coviddataset/Val'

            for i in os.listdir(Test_path):
                if i == 'Thumbs.db':
                    break
                img = image.load_img(Test_path+'//'+i,target_size=(224,224),interpolation='nearest')
    
                st.image(img)
      
                X = image.img_to_array(img)
    
                X = np.expand_dims(X,axis=0)
    
                images = np.vstack([X])
    
                Pred = model.predict(images)
    
                ind =int(Pred)
                y_out = Category[ind]
                st.write(f' PREDICTED OUTPUT:{y_out}' )


	 

with visual:
	st.header('VISUALIZATION USING GRAD-CAM')
	st.text('Visualization of the Images using GRAD-CAM heat-mapping')
	cov_col,nor_col = st.beta_columns(2)
	with cov_col:
            image = Image.open('Covid_cam.jpg')
            st.image(image, caption ='Covid X-ray image',width=300)
	with nor_col:
            image1 = Image.open('Normal_cam.jpg')
            st.image(image1, caption ='Normal X-ray image',width=300)
	
with graph:
	st.header('PERFORMANCE ANALYSIS')
	st.text('The performance metrics used for the model is Accuracy')
	Acc_col,Loss_col = st.beta_columns(2)
	with Acc_col:
            image2 = Image.open('Acc.png')
            st.image(image2 , captions='Accuracy graph of Train and Validate data',width=400)
            with Loss_col:
                image3 = Image.open('Loss.png')
                st.image(image3 , captions='Loss graph of Train and Validate data',width=400)

with safety:
    st.header('Safety measures to follow during the Covid-19 pandemic')
    st.markdown(' Wear a mask whenever you go out ')
    st.markdown(' Maintain 1 meter of social distance in the Public ')
    st.markdown(' Don\'t touch your nose and eyes ')
    st.markdown(' Wash your hands periodically with hand-sanitizer or soaps ')

with about:
    st.markdown(' # About')
    name_col,reg_col = st.beta_columns(2)
    with name_col:
        st.subheader('DHILIP KUMAR E')
        st.subheader('GOKUL M')
        st.subheader('JAGAN K')
        
    with reg_col:
        st.subheader('513417104013')
        st.subheader('513417104017')
        st.subheader('513417104302')
    guid_col,Hod_col = st.beta_columns(2)

    with guid_col:
        st.markdown('## GUIDE')
        st.subheader('Mr.M.Suresh.M.E.,')
        st.text('Teaching Fellow,Department of CSE')
    with Hod_col:
        st.markdown('## Head Of Department ')
        st.subheader('Dr.V.Kavitha.Ph.D.,')
        st.text('Head Of Department,CSE')
