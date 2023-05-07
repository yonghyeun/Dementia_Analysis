# 경로 처리
import streamlit as st
import os
from pathlib import Path
import glob

# 일반
import pandas as pd
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'
import koreanize_matplotlib
import numpy as np
import random
import tqdm as tqdm
import seaborn as sns

# 이미지 처리
import cv2
from PIL import Image, ImageOps
# import imgaug.augmenters as iaa

# 모델링
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import BatchNormalization
from keras.models import Model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import h5py
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# 모델
def build_model():
    input_img = Input(shape=(128, 128, 1), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    # x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    # x = Dropout(0.5, name='dropout2')(x)
    x = Dense(3, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model

model =  build_model()

model.load_weights("back.hdf5")

# streamlit 화면
st.title('치매 진단 AI 모델')
file = st.file_uploader('이미지를 올려주세요.', type=['jpg', 'png'])

if file is None:
    st.text('이미지를 먼저 올려주세요.')
    
# 뇌 MRI 사진인지 아닌지 판단 코드 넣기
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_resized = ImageOps.fit(image, (128, 128), Image.ANTIALIAS)
    img_resized = img_resized.convert('RGB')
    
    img_array = np.array(img_resized) # PIL Image 객체를 numpy 배열로 변환
    
    _, threshold_img = cv2.threshold(img_array, 145, 255, cv2.THRESH_BINARY) # numpy 배열을 cv2.threshold 함수에 전달
    
    canny_img = cv2.Canny(threshold_img,10,135)
    img_resized = np.array(canny_img)/255.
    
    y_pred = model.predict(img_resized.reshape([1, 128, 128]))

    y_pred_label = np.argmax(y_pred)
    
    if y_pred_label == 0: y_pred_label = '정상'
    elif y_pred_label == 1: y_pred_label = '경도 인지 장애'
    elif y_pred_label == 2: y_pred_label = '치매'
    
    result = f'{y_pred_label}'
    st.success(result)

