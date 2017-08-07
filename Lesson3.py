# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:40:38 2017

@author: zipliu
"""

import os, json
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from matplotlib import pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam

import importlib
import vgg16; importlib.reload(vgg16)
from vgg16 import Vgg16
import utils; importlib.reload(utils)

       
### Removing dropouts in VGG
# Parameters
path = 'V:/Data/FastAI/dogscats/sample/'
model_path = path + 'models/'
buildFeature = False
batch_size = 8
# Load VGG model with weights
vgg = Vgg16()
model = vgg.model
# get layers of VGG
layers = model.layers
## get last convolution model index
#last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D][-1]
## Get convolution layers and make them into a model
#conv_layers = layers[:last_conv_idx+1]
# Get first dense layer
first_dense_idx = [index for index, layer in enumerate(layers) if type(layer) is Dense][0]
# Get convolution layers till first Dense layers
conv_layers = layers[:first_dense_idx+1]
# Build conv model
conv_model = Sequential(conv_layers)
# Build or load features
if buildFeature:
    trn_batches = utils.get_batches(path+'train', shuffle=False, batch_size = batch_size)
    val_batches = utils.get_batches(path+'valid', shuffle=False, batch_size = batch_size)
    
    trn_classes = trn_batches.classes
    val_classes = val_batches.classes
    
    trn_labels = utils.onehot(trn_classes)
    val_labels = utils.onehot(val_classes)
    
    trn_features = conv_model.predict_generator(trn_batches, 
                                                trn_batches.n/trn_batches.batch_size)
    val_features = conv_model.predict_generator(val_batches, 
                                                val_batches.n/val_batches.batch_size)
    
    utils.save_array(model_path + 'train_convlayer_features.bc', trn_features)
    utils.save_array(model_path + 'valid_convlayer_features.bc', val_features)
    
    utils.save_array(model_path + 'train_convlayer_labels.bc', trn_labels)
    utils.save_array(model_path + 'valid_convlayer_labels.bc', val_labels)
    
else:
    trn_features = utils.load_array(model_path + 'train_convlayer_features.bc')
    val_features = utils.load_array(model_path + 'valid_convlayer_features.bc')

    trn_labels = utils.load_array(model_path + 'train_convlayer_labels.bc')
    val_labels = utils.load_array(model_path + 'valid_convlayer_labels.bc')
# Get fully-connected layers
#fc_layers = layers[last_conv_idx+1:]
fc_layers = layers[first_dense_idx+1:]
# Build FC layers without drop out
fc_model = Sequential([                                              \
#        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]), \
#        Flatten(),                                                  \
#        Dense(4096, activation='relu'),                             \
        Dropout(0.0, input_shape=conv_layers[-1].output_shape[1:]), \
        Dense(4096, activation='relu'),                             \
        Dropout(0.0),                                               \
        Dense(2, activation='softmax')                              \
        ])
# Scale initial weights from VGG16 model by 0.5, because we remove dropout
for layer1, layer2 in zip(fc_model.layers, fc_layers):
    #if type(layer1) is Dense and layer1.output_shape[1] != 2:  
    if layer1.output_shape[1] != 2:
        layer1.set_weights(utils.prorate_weight(layer2, 0.5))
# Compile model
fc_model.compile(optimizer=RMSprop(lr=0.00001, rho=0.7), loss='categorical_crossentropy', \
                 metrics=['accuracy'])
# Fit the model using train and valid data
fc_model.fit(trn_features, trn_labels, epochs=3, batch_size=batch_size, \
             validation_data=(val_features, val_labels))
# Save weights 
fc_model.save_weights(model_path + 'vgg16_dropout0.h5')

### Data augmentation illustration
# Parameters
path = 'V:/Data/FastAI/dogscats/sample/'
K.set_image_dim_ordering('tf')
# Create image generator with augmentation
gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1,        \
                               height_shift_range=0.1,                          \
                               shear_range=0.15, zoom_range=0.1,                \
                               channel_shift_range=10, horizontal_flip=True)
# Get image
raw = ndimage.imread(path+'train/cats/cat.394.jpg')
img = np.expand_dims(raw, 0)
# Create generator iterator 
aug_iter = gen.flow(img)
# Get eight randomly augmented images
aug_imgs = [next(aug_iter)[0].astype(np.uint8) for i in range(8)]
# Display original image
plt.imshow(img[0])
# Augmented images
utils.plots(aug_imgs, (20,8), rows=2)

### Add data augmentation in VGG
# Local method
def divide_vgg_model():
    """
    Divide VGG model into convolution model (from Lamda till first Dense layer)
    and fully-connected model (from first drop-out to end). The convolution model 
    retains all VGG layers and weights; the FC model will have drop-out removed
    and change output-shape to 2. This routine only construct the models, but 
    doesn't compile them
    """
    # Load VGG model with weights
    vgg = Vgg16()
    model = vgg.model
    # get layers of VGG
    layers = model.layers
    # Get first dense layer
    first_dense_idx = [index for index, layer in enumerate(layers) if type(layer) is Dense][0]
    # Get convolution layers till first Dense layers
    conv_layers = layers[:first_dense_idx+1]
    #fc_layers = layers[last_conv_idx+1:]
    fc_layers = layers[first_dense_idx+1:]
    # Build conv model
    conv_model = Sequential(conv_layers)
    # build FC model
    fc_model = Sequential([                                              \
    #        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]), \
    #        Flatten(),                                                  \
    #        Dense(4096, activation='relu'),                             \
            Dropout(0.0, input_shape=conv_layers[-1].output_shape[1:]), \
            Dense(4096, activation='relu'),                             \
            Dropout(0.0),                                               \
            Dense(2, activation='softmax')                              \
            ])
    # Scale initial weights from VGG16 model by 0.5, because we remove dropout
    for layer1, layer2 in zip(fc_model.layers, fc_layers):
        #if type(layer1) is Dense and layer1.output_shape[1] != 2:  
        if layer1.output_shape[1] != 2:
            layer1.set_weights(utils.prorate_weight(layer2, 0.5))

    return (conv_model, fc_model)

# Parameters
path = 'V:/Data/FastAI/dogscats/sample/'
model_path = path + 'models/'
K.set_image_dim_ordering('th')
batch_size = 4
# Create image generator with augmentation options
gen_aug = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,    \
                               height_shift_range=0.1, zoom_range=0.1,          \
                               horizontal_flip=True)
# Get training batch using generator with augmentation
trn_batches = utils.get_batches(path+'train/', gen_aug, batch_size=batch_size)
# Get valid batch using default generator without augmentation
#!!! NB: we don't want augmentation or shuffle on the validation set
val_batches = utils.get_batches(path+'valid/', shuffle=False, batch_size=batch_size)
# Get two sub-models from VGG
conv_model, fc_model = divide_vgg_model()
# Make conv layers non-trainable
for layer in conv_model.layers:
    layer.trainable = False
# Connect fc_model to conv_model
conv_model.add(fc_model)
# Compile new model
conv_model.compile(optimizer=RMSprop(lr=0.0001, rho=0.7), loss='categorical_crossentropy',\
                   metrics=['accuracy'])
# Use generator to fit the model
conv_model.fit_generator(trn_batches, steps_per_epoch=trn_batches.n/trn_batches.batch_size, \
                         epochs=3, validation_data=val_batches, \
                         validation_steps=val_batches.n/val_batches.batch_size)
# Save weights
conv_model.save_weights(model_path + 'vgg_dropout0_aug1.h5')


### Add batch normalization
# Parameters
path = 'V:/Data/FastAI/dogscats/sample/'
model_path = path + 'models/'
K.set_image_dim_ordering('th')
batch_size = 4
# Create image generator with augmentation options
gen_aug = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,    \
                               height_shift_range=0.1, zoom_range=0.1,          \
                               horizontal_flip=True)
# Get training batch using generator with augmentation
trn_batches = utils.get_batches(path+'train/', gen_aug, batch_size=batch_size)
# Get valid batch using default generator without augmentation
#!!! NB: we don't want augmentation or shuffle on the validation set
val_batches = utils.get_batches(path+'valid/', shuffle=False, batch_size=batch_size)
# Local method for separating VGG into convolution and FC layers
def separate_vgg_model():
    """
    Separate VGG model into convolution layers (from Lamda till first Dense layer)
    and fully-connected layers (from first drop-out to end). 
    """
    # Load VGG model with weights
    vgg = Vgg16()
    model = vgg.model
    # get layers of VGG
    layers = model.layers
    # Get first dense layer
    first_dense_idx = [index for index, layer in enumerate(layers) if type(layer) is Dense][0]
    # Get convolution layers till first Dense layers
    conv_layers = layers[:first_dense_idx+1]
    #fc_layers = layers[last_conv_idx+1:]
    fc_layers = layers[first_dense_idx+1:]
    # return
    return (conv_layers, fc_layers)
# Get conv_layers and lock them
conv_layers,_ = separate_vgg_model()
for layer in conv_layers:
    layer.trainable = False
# Build new FC layers
fc_model = Sequential([ \
                       BatchNormalization(input_shape=conv_layers[-1].output_shape[1:]), \
                       Dropout(0.75), \
                       Dense(4096, activation='relu'), \
                       BatchNormalization(),    \
                       Dropout(0.75), \
                       Dense(2, activation='softmax')                              \
                      ])
# Build final model
final_model = Sequential(conv_layers)
for layer in fc_model.layers:
    final_model.add(layer)
# Compile model
final_model.compile(optimizer=Adam(), loss='categorical_crossentropy', \
                    metrics=['accuracy'])
# Fit model using generated batches
final_model.fit_generator(trn_batches, steps_per_epoch=trn_batches.samples/trn_batches.batch_size, \
                          epochs=3, validation_data=val_batches,
                          validation_steps=val_batches.samples/val_batches.batch_size)
# Save model weights
final_model.save_weights(model_path+'vgg16_dropout0.75_aug_bn.h5')

