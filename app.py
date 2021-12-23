import streamlit as st
import os
import tensorflow as tf
import numpy as np
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

st.title('COVID Detection Model')
Category =['COVID','NORMAL']

st.text('Upload the image')
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
    
        
    



