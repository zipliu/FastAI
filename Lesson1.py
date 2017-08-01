# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:50:09 2017

Re-implementation of Fast AI 1, Lesson 1 class project
Dogs vs Cats

@author: liu.zhipeng
"""

# Imports
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import importlib
import vgg16; importlib.reload(vgg16)
from vgg16 import Vgg16
import utils

# Data path
# Inside the path, there shall be train and valid folder
# Inside each folder, there shall be one sub-folder for each category
path = 'V:/Data/FastAI/dogscats/sample/'

# Fine-tune VGG model
batch_size = 4
vgg = Vgg16()
train_batches = vgg.get_batches(path + 'train', batch_size = batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size = batch_size * 2)
vgg.finetune(train_batches)
vgg.fit(train_batches, val_batches, nb_epoch = 1)


# Validate new model on a different dataset
test_batches = vgg.get_batches('V:/Data/FastAI/dogscats/valid', batch_size = batch_size * 2);
imgs, labels = next(test_batches)
utils.plots(imgs, titles=labels)
scores, idxs, classes = vgg.predict(imgs)
for i in range(len(scores)):
    print('Score: {}, ID: {}, Class: {}'.format(scores[i], idxs[i], classes[i]))