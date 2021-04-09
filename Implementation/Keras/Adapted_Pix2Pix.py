# -*- coding: utf-8 -*-
#Adapted from:
#https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/


from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import numpy as np
import os
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.initializers import Zeros
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.preprocessing.image import load_img, save_img
import keras
import time
from keras import backend as K

input_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/backprojected_dataset_png"
target_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/full_year_project_data_png"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") 
    ]
)

def normalize_images(image_in):
    image_out=(image_in-127.5)/127.5
    return image_out

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.002)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(mean=-0.5, stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(mean=-0.5, stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(512,512,1)):
	# weight initialization
	init = RandomNormal(mean=-0.5, stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = X1/255 #(X1 - 127.5) / 127.5
	X2 = X2/255 #(X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(idx,n_samples, patch_shape,input_img_paths,target_img_paths):
    	# generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    batch_size=n_samples
    i = idx * n_samples
    img_res=512
    img_size = (512, 512)
    batch_input_img_paths = input_img_paths[i : i + batch_size]
    batch_target_img_paths = target_img_paths[i : i + batch_size]
    X1 = np.zeros((batch_size,img_res,img_res,1), dtype="float32")
    for j, path in enumerate(batch_input_img_paths):
        img = load_img(path, target_size=img_size, color_mode="grayscale")
        imx=normalize_images(np.asarray_chkfinite(img, dtype=float))
        #print(path)
        X1[j,:,:,0] = imx
    X2 = np.zeros((batch_size,img_res,img_res,1), dtype="float32")
    for j, path in enumerate(batch_target_img_paths):
        img = load_img(path, target_size=img_size, color_mode="grayscale")
        imy=normalize_images(np.asarray_chkfinite(img, dtype=float))
        #print(path)
        X2[j,:,:,0] = imy
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# train pix2pix models
def train(d_model, g_model, gan_model, input_img_paths,target_img_paths, n_batch=4, n_epochs=5):
	# determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	# unpack dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(5063 / n_batch)
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
    start=time.time()     
    
    for e in range(n_epochs):
        savename=r'C:\Users\darth\Desktop\Final Year Project\GPU Training\GAN\Generator_prog\CurrentGen'+str(e)+'.png'
        for j in range(bat_per_epo):
            i=int(j+(e*bat_per_epo))
    		# select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(j,n_batch, n_patch,input_img_paths,target_img_paths)
            		# generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            		# update discriminator for real samples
            save_img(savename,X_fakeB[0,:,:,:] ,target_size=(512,512), color_mode="grayscale")
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            		# update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            		# update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real,X_realB ])
            		# summarize performance
            print('>E: %d I: %d, d1[%.3f] d2[%.3f] g[%.3f] | Time: %f s' % (e+1,i+1, d_loss1, d_loss2, g_loss,(time.time()-start))
                  )
        if (e % 10)==0:    
            print('Saving..')
            gan_model.save(r'C:\Users\darth\Desktop\Final Year Project\GPU Training\GAN\GAN_Check_Ep'+str(e)+'.h5')
    
    gan_model.save(r'C:\Users\darth\Desktop\Final Year Project\GPU Training\GAN\GAN_Check_Ep'+str(e)+'.h5')
        

# load image data
# define the models
image_shape=(512,512,1)
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
gan_model.summary()
keras.utils.plot_model(gan_model, to_file="GANKERAS.png", show_shapes=True)
keras.utils.plot_model(g_model, to_file="GENKERAS.png", show_shapes=True)
keras.utils.plot_model(d_model, to_file="DISCKERAS.png", show_shapes=True)
# train model
train(d_model, g_model, gan_model,input_img_paths,target_img_paths)
