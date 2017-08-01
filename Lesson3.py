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

### Removing dropouts
# Parameters
path = 'V:/Data/FastAI/dogscats/sample/'
model_path = path + 'models/'
buildFeature = False
batch_size = 4
# Load VGG model with weights
vgg = Vgg16()
model = vgg.model
# get layers of VGG
layers = model.layers
# get last convolution model index
last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D][-1]
# Get convolution layers and make them into a model
conv_layers = layers[:last_conv_idx+1]
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
fc_layers = layers[last_conv_idx+1:]
# Build FC layers without drop out
fc_model = Sequential([                                              \
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]), \
        Flatten(),                                                  \
        Dense(4096, activation='relu'),                             \
        Dropout(0.0),                                               \
        Dense(4096, activation='relu'),                             \
        Dropout(0.0),                                               \
        Dense(2, activation='softmax')                              \
        ])
# Scale initial weights from VGG16 model by 0.5, because we remove dropout
for layer1, layer2 in zip(fc_model.layers, fc_layers):
    if type(layer1) is Dense and layer1.output_shape[1] != 2:  
        layer1.set_weights(utils.prorate_weight(layer2, 0.5))
# Compile model
fc_model.compile(optimizer=RMSprop(lr=0.00001, rho=0.7), loss='categorical_crossentropy', \
                 metrics=['accuracy'])
# Fit the model using train and valid data
fc_model.fit(trn_features, trn_labels, epochs=3, batch_size=batch_size, \
             validation_data=(val_features, val_labels))
# Save weights 
fc_model.save_weights(model_path + 'vgg16_dropout0.h5')
