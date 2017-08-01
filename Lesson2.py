# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:25:42 2017

@author: liu.zhipeng
"""

import os, json
from glob import glob
import numpy as np
from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from keras import backend as K
# Set image dimension in Theano order
K.set_image_dim_ordering('th')  
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

import importlib
import utils; importlib.reload(utils)
import vgg16; importlib.reload(vgg16)
from vgg16 import Vgg16




### Linear model
#x = random((30,2))
#y = np.dot(x, [2., 3.]) + 1.
#lm = Sequential([Dense(1, input_shape=(2,))])
#lm.compile(optimizer=SGD(lr=0.1), loss='mse')
#print('Before optimization fitting error: {}\n'.format(lm.evaluate(x, y, verbose=0)))
#
#lm.fit(x, y, epochs = 50, batch_size=1)
#print('After optimization fitting error: {}\n'.format(lm.evaluate(x, y, verbose=0)))
#print('Weights:')
#lm.get_weights()


### 1 Dense Layer after VGG-16 Outputs
# parameters
bReloadData = False
bReloadLabel = False
bReloadFeature = False
bReloadModel = False
batch_size = 10
epoch_cnt = 1
view_cnt = 4
path = 'V:/Data/FastAI/dogscats/sample/'
model_path = path + 'models/'
if not os.path.exists(model_path): 
    os.mkdir(model_path)

# Load images
if not bReloadData: 
    trn_data = utils.get_data(path+'train')
    val_data = utils.get_data(path+'valid')
 
    utils.save_array(model_path+'train_data.bc', trn_data)
    utils.save_array(model_path+'valid_data.bc', val_data)
else:
    trn_data = utils.load_array(model_path+'train_data.bc')
    val_data = utils.load_array(model_path+'valid_data.bc')
    
    
# Load labels
if not bReloadLabel:
    trn_batch = utils.get_batches(path+'train', shuffle=False, batch_size=1)
    val_batch = utils.get_batches(path+'valid', shuffle=False, batch_size=1)
    
    trn_classes = trn_batch.classes
    val_classes = val_batch.classes
    
    trn_labels = utils.onehot(trn_classes)
    val_labels = utils.onehot(val_classes)

    utils.save_array(model_path+'train_label.bc', trn_labels)
    utils.save_array(model_path+'valid_label.bc', val_labels)
else:
    trn_labels = utils.load_array(model_path+'train_label.bc')
    val_labels = utils.load_array(model_path+'valid_label.bc')
    


# Load VGG model and use its prediction as features
if not bReloadFeature:
    vgg = Vgg16()
    trn_features = vgg.model.predict(trn_data, batch_size)
    val_features = vgg.model.predict(val_data, batch_size)
    
    utils.save_array(model_path + 'train_vgg_feature.bc', trn_features)
    utils.save_array(model_path + 'valid_vgg_feature.bc', val_features)
else:
    trn_features = utils.load_array(model_path + 'train_vgg_feature.bc')
    val_features = utils.load_array(model_path + 'valid_vgg_feature.bc')
    
# Use linear-model on VGG predictions
lm = Sequential([Dense(2, activation='softmax', input_shape=(1000,))])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
if not bReloadModel:
    lm.fit(trn_features, trn_labels, epochs=epoch_cnt, batch_size=batch_size, 
           validation_data=(val_features, val_labels))
    lm.save_weights(model_path + 'Dense1.h5')
else:
    lm.load_weights(model_path + 'Dense1.h5')
lm.summary()

# Check Results
preds = lm.predict_classes(val_features, batch_size=batch_size) # classes
probs = lm.predict_proba(val_features, batch_size)  # probabilities    

val_batch = utils.get_batches(path+'valid', shuffle=False, batch_size=1)
val_classes = val_batch.classes
filenames = val_batch.filenames

#1. A few correct predictions at random
correct = np.where(preds==val_labels[:,1])[0]
idx = permutation(correct)[:view_cnt]
if len(idx)==view_cnt: 
    utils.plots_idx(path+'valid/', filenames, idx, np.rint(probs[idx,:]*100))

#2. A few incorrect labels at random
incorrect = np.where(preds != val_labels[:,1])[0]
idx = permutation(incorrect)[:view_cnt]
if len(idx)==view_cnt: 
    utils.plots_idx(path+'valid/', filenames, idx, np.rint(probs[idx,:]*100))

#3. The images we most confident were cats, and are actually cats
correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
most_correct_cats = np.argsort(probs[correct_cats,0])[::-1][:view_cnt]
if len(most_correct_cats)==view_cnt: 
    utils.plots_idx(path+'valid/', filenames, correct_cats[most_correct_cats], 
                    np.rint(probs[correct_cats[most_correct_cats],:]*100))

#4. The images we most confident were dogs, and are actually dogs
correct_dogs = np.where((preds==1) & (preds==val_labels[:,1]))[0]
most_correct_dogs = np.argsort(probs[correct_dogs,1])[::-1][:view_cnt]
if len(most_correct_dogs)==view_cnt: 
    utils.plots_idx(path+'valid/', filenames, correct_dogs[most_correct_dogs], 
                    np.rint(probs[correct_dogs[most_correct_dogs], :]*100))

#5. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((preds==0) & (val_labels[:,1]==1))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats,0])[::-1][:view_cnt]
if len(most_incorrect_cats)==view_cnt:
    utils.plots_idx(path+'valid/', filenames, incorrect_cats[most_incorrect_cats],
                    np.rint(probs[incorrect_cats[most_incorrect_cats], :]*100))
    
#6. The images we were most confident were dogs, but are actually dogs
incorrect_dogs = np.where((preds==1) & (val_labels[:,1]==0))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs,1])[::-1][:view_cnt]
if len(most_incorrect_dogs)==view_cnt:
    utils.plots_idx(path+'valid/', filenames, incorrect_dogs[most_incorrect_dogs],
                    np.rint(probs[incorrect_dogs[most_incorrect_dogs],:]*100))
    
#7. the most uncertain labels
most_uncertain = np.argsort(np.abs(probs[:,0]-0.5))[:view_cnt]
if len(most_uncertain)==view_cnt:
    utils.plots_idx(path+'valid/', filenames, most_uncertain, 
                    np.rint(probs[most_uncertain,:]*100))
    
# Confusion matrix
c_matrix = confusion_matrix(val_classes, preds)
utils.plot_confusion_matrix(c_matrix, val_batch.class_indices)

### Re-structure VGG-16 model 
# parameters
path = 'V:/Data/FastAI/dogscats/sample/'
batch_size = 4
# Load pre-trained Vgg16 model
vgg = Vgg16()
# Get model
model = vgg.model
model.summary()
# Remove last layer
model.pop()     # pop last layer
# Make the rest layers un-trainable
for layer in model.layers:
    layer.trainable = False
# Add a new final layer
model.add(Dense(2, activation='softmax'))
# Compile model
opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Get training and validation data
model_path = path + 'models/'
val_data = utils.load_array(model_path+'valid_data.bc')
val_labels = utils.load_array(model_path+'valid_label.bc')
trn_data = utils.load_array(model_path+'train_data.bc')
trn_labels = utils.load_array(model_path+'train_label.bc')
gen = image.ImageDataGenerator()
trn_batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)
model.fit_generator(trn_batches, samples_per_epoch=trn_batches.n, nb_epoch=1,
                    validation_data=val_batches, nb_val_samples=val_batches.n)
#trn_batch2 = utils.get_batches(path+'train', shuffle=False, batch_size=batch_size)
#val_batch2 = utils.get_batches(path+'valid', shuffle=False, batch_size=batch_size)
#model.fit_generator(trn_batch2, samples_per_epoch=trn_batch2.samples, nb_epoch=1,
#                    validation_data=val_batch2, nb_val_samples=val_batch2.samples)
model.evaluate(val_data, val_labels)



### Re-train all Dense layers of VGG 16
# Get layers of pre-trained model
layers = model.layers
# Get the index of the 1st dense layer
first_dense_idx = [index for index, layer in enumerate(layers) if type(layer) is Dense][0]
# Make first denser layer and its succeeding layers trainable
for layer in layers[first_dense_idx:]:
    layer.trainable = True
# Because we did not change the model, there no need to "compile"
# Set learning rate
opt = RMSprop(lr=0.1)
K.set_value(opt.lr, 0.01)
# Re-train the model
batch_size = 4
trn_batches = utils.get_batches(path+'train/', shuffle=True, batch_size=batch_size)
val_batches = utils.get_batches(path+'valid/', shuffle=True, batch_size=batch_size)
