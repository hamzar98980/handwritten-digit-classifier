import streamlit as slt
from PIL import Image
# from tensorflow.keras import keras
from tensorflow.keras.models import load_model
import numpy as np


slt.title('Upload Handwritten Number Image')

upload_file = slt.file_uploader("Upload Image")

if upload_file is not None:

    img = Image.open(upload_file)
    model = load_model('digit_handwritten_model.keras')
    
    greyscaleImg = img.convert('L')
    greyscaleImgArr = np.array(greyscaleImg) / 255
    greyscaleImgArr = greyscaleImgArr.reshape(1,784)
    predicited = model.predict(greyscaleImgArr).argmax()
    print(predicited)
    print(model.predict(greyscaleImgArr))
    slt.image(image=img,width=150)

    slt.title(f'Predicition: {predicited}')
    # slt.write(predicited)

    # slt.image(image=greyscaleImgArr,width=150)
