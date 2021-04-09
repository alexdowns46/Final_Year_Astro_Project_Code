# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:05:53 2021

@author: darth
"""

import numpy as np
import os
import math
import time
import skimage.metrics as immetrics

from numpy.linalg import norm
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from tensorflow.keras import layers
from tensorflow.image import ssim
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras as keras
num_channels=56
img_size = (512, 512)

dirty_dir = "C:/Users/darth/Desktop/Final Year Project/matlab_files/matlab_files/Validation data/backprojected_dataset"
clean_dir = "C:/Users/darth/Desktop/Final Year Project/matlab_files/matlab_files/Validation data/clean_dataset"

#dirty_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/backprojected_dataset_png"
#clean_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/full_year_project_data_png"

dirty_img_paths = sorted(
    [
        os.path.join(dirty_dir, fname)
        for fname in os.listdir(dirty_dir)
        if fname.endswith(".png")
    ]
)

clean_img_paths = sorted(
    [
        os.path.join(clean_dir, fname)
        for fname in os.listdir(clean_dir)
        if fname.endswith(".png")
    ]
)

#Define some loss functions and metrics
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303





# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


def normalize_images(image_in):
    image_out=(image_in-127.5)/127.5
    return image_out

def psnr(img1, img2,Pmax):

    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(Pmax / math.sqrt(mse))

def snr(img1,img2):
    snrout=20*np.log10((norm(img1))/norm(img1-img2))
    return snrout

def GetMetrics(ref_img,test_img):
    #Normalize images to [0,1]
    ref_img=(ref_img+1)/2
    test_img=(test_img+1)/2
    #Get metrics
    PSNR=psnr(ref_img,test_img,1)
    SNR=snr(ref_img,test_img)
    SSIM=immetrics.structural_similarity(ref_img,test_img,data_range=1)
  
    return PSNR, SNR, SSIM

#kmodel.load_weights('UNeT_RDBcluster_TEST.h5')
ganmodel=keras.models.load_model('GAN_Check_Ep449.h5',custom_objects={'root_mean_squared_error': root_mean_squared_error,'PSNR': PSNR})
ganmodel.summary()


kmodel=ganmodel.layers[1]
kmodel.summary()


meanSNR=0
meanSSIM=0
meanPSNR=0

SNRArr=np.zeros(100)
PSNRArr=np.zeros(100)
SSIMArr=np.zeros(100)
TimeArr=np.zeros(100)

outdir=r'C:\Users\darth\Desktop\Final Year Project\GPU Training\GAN\GAN_Reconstructions'
for i in range(100):

    img_clean = load_img(clean_img_paths[i], target_size=img_size, color_mode="grayscale")
    ar_clean=normalize_images(img_to_array(img_clean))
    
    img_dirty= load_img(dirty_img_paths[i], target_size=img_size, color_mode="grayscale")
    ar_dirty=normalize_images(img_to_array(img_dirty))
    
    net_input = np.zeros((1,512,512,1), dtype="float32")
    net_match = np.zeros((1,512,512,1), dtype="float32")
    net_input[0,:,:,:]=ar_dirty;
    net_match[0,:,:,:]=ar_clean;
    start=time.time()
    net_out=kmodel.predict(x=net_input, batch_size=1)
    TimeArr[i]=start-time.time()
    ar_out=np.zeros((512,512), dtype="float")
    ar_out=(net_out[0,:,:,0])
    
    PSNR, SNR, SSIM = GetMetrics(ar_clean[:,:,0],ar_out[:,:])
    
    SSIMArr[i]=SSIM
    SNRArr[i]=SNR
    PSNRArr[i]=PSNR
    
    Temp=clean_img_paths[i].replace(clean_dir,'')
    ImName=Temp[1:]
    savename=outdir+'/'+ ImName + '_PSNR_' + str(PSNR) +'dB_SNR_' + str(SNR) +'dB_SSIM_'+ str(SSIM)+ '.png' 
    print(savename)
    #save_img(savename,net_out[0,:,:,:] ,target_size=(512,512), color_mode="grayscale")

MeanData="PSNR Avg: " + str(np.mean(PSNRArr)) + " | SNR Avg: " + str(np.mean(SNRArr)) + " | SSIM Avg: " + str(np.mean(SSIMArr)) + " | Time Avg: " + str(np.mean(TimeArr)) + " | PSNR Std: " + str(np.std(PSNRArr)) + " | SNR Std: " + str(np.std(SNRArr)) + " | SSIM Std: " + str(np.std(SSIMArr)) + " | Time Std: " + str(np.std(TimeArr))

print(MeanData)

plt.figure(1)
cleanplot = plt.imshow(ar_clean)
plt.figure(2)
dirtyplot = plt.imshow(ar_dirty)
plt.figure(3)
outplot=plt.imshow(ar_out)
#save_img("/mnt/shared/home/ahd1/TF_Training/TEST.png",ar_out)


















