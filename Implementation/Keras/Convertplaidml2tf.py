# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:10:53 2021

@author: darth
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import visualkeras
num_channels=56
 
def RDB(inputlayer,channels):
    
	concatp1=layers.Conv2D(channels,1, padding="same")(inputlayer)
	x = layers.Conv2D(channels,3, padding="same")(concatp1)
	concatp2 = layers.Activation("relu")(x)
	x = layers.concatenate([concatp2, concatp1],axis=3) #Attach residual concat
	
	x = layers.Conv2D(channels,3, padding="same")(x)
	concatp3 = layers.Activation("relu")(x)
	x = layers.concatenate([concatp3, concatp1],axis=3) #Attach residual concat
	x = layers.concatenate([x, concatp2],axis=3) #Attach residual concat
	
	x = layers.Conv2D(channels,3, padding="same")(x)
	concatp4 = layers.Activation("relu")(x)
	x = layers.concatenate([concatp4, concatp1],axis=3) #Attach residual concat
	x = layers.concatenate([x, concatp2],axis=3) #Attach residual concat
	x = layers.concatenate([x, concatp3],axis=3) #Attach residual concat
	
	x = layers.Conv2D(channels,1, padding="same")(x)
	
	x=layers.Add()([x,concatp1])
	
	return x

def get_model_rdb(img_size, num_classes):
    inputs = keras.Input(shape=(512,512,1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x =RDB(inputs,num_channels)
    concat1 =  layers.Activation("relu")(x) 
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat1)  # Downsample!


    #Depth 2
    x =RDB(x,num_channels*2)
    concat2 = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat2)  # Downsample!

    #Depth 3
    x =RDB(x,num_channels*4)
    
    x = layers.Conv2DTranspose(num_channels*4,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    #x = layers.UpSampling2D(size=(2, 2))(x)

    #Back to depth 2
    x = layers.concatenate([x, concat2],axis=3) #Attach residual concat
    x =RDB(x,num_channels*2)
    
    #x = layers.UpSampling2D(size=(2, 2))(x)   
    x = layers.Conv2DTranspose(num_channels*2,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    
    #Back to depth 1
    x = layers.concatenate([x, concat1],axis=3) #Attach residual concat
    x =RDB(x,num_channels)

    outputs = layers.Conv2D(1,1, padding="same")(x)    
      
    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def CBNR(inputlayer,channels):
    x = layers.Conv2D(channels,3, padding="same")(inputlayer)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(channels,3, padding="same")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(channels,3, padding="same")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
   #x=layers.Add()([x,concatp1])
	
    return x

def get_model_add(img_size, num_classes):
    inputs = keras.Input(shape=(512,512,1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x =CBNR(inputs,num_channels)
    concat1 =  layers.Activation("relu")(x) 
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat1)  # Downsample!


    #Depth 2
    x =CBNR(x,num_channels*2)
    concat2 = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat2)  # Downsample!

    #Depth 3
    x =CBNR(x,num_channels*4)
    
    x = layers.Conv2DTranspose(num_channels*4,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    #x = layers.UpSampling2D(size=(2, 2))(x)

    #Back to depth 2
    x = layers.concatenate([x, concat2],axis=3) #Attach residual concat
    x =CBNR(x,num_channels*2)
    
    #x = layers.UpSampling2D(size=(2, 2))(x)   
    x = layers.Conv2DTranspose(num_channels*2,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    
    #Back to depth 1
    x = layers.concatenate([x, concat1],axis=3) #Attach residual concat
    x =CBNR(x,num_channels)
    
    x = layers.Conv2D(1,1, padding="same")(x)    
    outputs=layers.Add()([x,inputs])
     
    # Define the model
    model = keras.Model(inputs, outputs)
    return model

#Define U-Net   
def get_model_basic(img_size, num_classes):
    inputs = keras.Input(shape=(512,512,1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x =CBNR(inputs,num_channels)
    concat1 =  layers.Activation("relu")(x) 
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat1)  # Downsample!


    #Depth 2
    x =CBNR(x,num_channels*2)
    concat2 = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(concat2)  # Downsample!

    #Depth 3
    x =CBNR(x,num_channels*4)
    
    x = layers.Conv2DTranspose(num_channels*4,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    #x = layers.UpSampling2D(size=(2, 2))(x)

    #Back to depth 2
    x = layers.concatenate([x, concat2],axis=3) #Attach residual concat
    x =CBNR(x,num_channels*2)
    
    #x = layers.UpSampling2D(size=(2, 2))(x)   
    x = layers.Conv2DTranspose(num_channels*2,1,strides=(2,2),data_format="channels_last",padding="same")(x)  # Upsample!
    
    #Back to depth 1
    x = layers.concatenate([x, concat1],axis=3) #Attach residual concat
    x =CBNR(x,num_channels)

    outputs = layers.Conv2D(1,1, padding="same")(x)    
      
    # Define the model
    model = keras.Model(inputs, outputs)
    return model
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
kmodel = get_model_add((512,512), 1)

kmodel.load_weights('UNeT_Add_E45.h5')
#tf.keras.models.l_model(kmodel,'UNET_TRAIN_14EPOCH.h5')

with tf.compat.v1.Session() as sess:

  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
  print(result)
  
  
tf.keras.models.save_model(kmodel,'TF_UNET_ADD_E45.h5')
kmodel.summary()
visualkeras.layered_view(kmodel, to_file='output.png').show()
weights, biases = kmodel.layers[1].get_weights()
#UNeT_RDB_E45