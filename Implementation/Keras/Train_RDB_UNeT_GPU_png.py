import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras import layers
import keras
from keras.preprocessing.image import load_img
from keras import backend as K

print("Load")
input_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/backprojected_dataset_robust_png"
target_dir = "C:/Users/darth/Desktop/Final Year Project/Files/datasets/full_year_project_data_png"
img_size = (512, 512)
img_res=512
batch_size = 1
num_channels=56

#Define some loss functions and metrics
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

#Load dataset
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

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
# Define RDB
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


#Define U-Net   
def get_model(img_size, num_classes):
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
    
    x = layers.Conv2DTranspose(num_channels*4,1,strides=(2,2),data_format="channels_last")(x)  # Upsample!
    #x = layers.UpSampling2D(size=(2, 2))(x)

    #Back to depth 2
    x = layers.concatenate([x, concat2],axis=3) #Attach residual concat
    x =RDB(x,num_channels*2)
    
    #x = layers.UpSampling2D(size=(2, 2))(x)   
    x = layers.Conv2DTranspose(num_channels*2,1,strides=(2,2),data_format="channels_last")(x)  # Upsample!
    
    #Back to depth 1
    x = layers.concatenate([x, concat1],axis=3) #Attach residual concat
    x =RDB(x,num_channels)

    outputs = layers.Conv2D(1,1, padding="same")(x)    
      
    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
#kmodel = get_model(img_size, 1)



#kmodel = keras.models.load_model('UNET_TRAIN_1_5_e.h5')

def normalize_images(image_in):
    image_out=image_in/255
    return image_out

def normalize_images_OLD(image_in):
    min_val=np.amin(image_in)
    image_in=image_in-min_val
    max_val=np.amax(image_in)
    image_out=image_in/max_val
    return image_out


class ImageLoad(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,img_res,img_res,1), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            imx=normalize_images(np.asarray_chkfinite(img, dtype=float))
            x[j,:,:,0] = imx
        y = np.zeros((self.batch_size,img_res,img_res,1), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            imy=normalize_images(np.asarray_chkfinite(img, dtype=float))
            y[j,:,:,0] = imy
        return x, y
    
import random

# Split our img paths into a training and a validation set
val_samples = 300
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = ImageLoad(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = ImageLoad(batch_size, img_size, val_input_img_paths, val_target_img_paths)

optimizer = keras.optimizers.Adam(lr=0.0009)

 #DONT COMPILE MODEL AGAIN IF RESUMING TRAINING!
Resume=True
Recompile=False

if Resume==True:
    kmodel=keras.models.load_model('UNeT_RDB_robust_5e_E50.h5',custom_objects={'root_mean_squared_error': root_mean_squared_error,'PSNR': PSNR}) #Load model
    if Recompile==True:
            kmodel.compile(optimizer=optimizer, loss=root_mean_squared_error,metrics =["mean_absolute_error",PSNR])
else:
    kmodel = get_model(img_size, 1)
    kmodel.compile(optimizer=optimizer, loss=root_mean_squared_error,metrics =["mean_absolute_error",PSNR])
    
keras.utils.plot_model(kmodel, to_file="UNETKERAS.png", show_shapes=True)
kmodel.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("Unet_RDB_robust_Checkpoint.h5",save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 10

kmodel.fit_generator(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#kmodel.fit_generator(train_gen, epochs=epochs, callbacks=callbacks)
weights, biases = kmodel.layers[1].get_weights()
kmodel.save("UNeT_RDB_robust_5e_E55.h5")


















