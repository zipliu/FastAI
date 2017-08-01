import numpy as np
from matplotlib import pyplot as plt
import bcolz
from keras import backend as K
# Set image dimension in Theano order
K.set_image_dim_ordering('th')  
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
import itertools
from itertools import chain


### Plots & Figures
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    """
    plots(ims, figsize=(12,6), rows=1, interp=False, titles=None)
    plot a list of images
    """
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def plots_idx(path, filenames, idx, titles=None):
    """
    plots_idx(path, filenames, idx, titles=None):
    Given path, all file names in the path, and idx, plots images indexed by idx
    """
    if len(idx) != 0 : plots([image.load_img(path+filenames[k]) for k in idx], titles = titles)
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def close_all():
    plt.close("all")
    
### Save and Load numpy array
def save_array(fname, arr):
    """
    save_array(fname, arr)
    Save numpy NDARRAY to BCOLZ file
    """
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
def load_array(fname):
    """ 
    load_array(fname)
    Load numpy NDARRAY from BCOLZ file 
    """
    return bcolz.open(fname)

### Get image batches
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4,
                class_mode='categorical', target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size, class_mode=class_mode,
                                   shuffle=shuffle, batch_size=batch_size)
    
def get_data(path, target_size=(224,224)):
    """ Given a path, return images in numpy NDARRAY of [#, RGB, H, W] dimension"""
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None,
                          target_size=target_size);
    return np.concatenate([batches.next() for i in range(batches.samples)])

### One-hot encoder
def onehot(x):
    """ 
    onehot(x)
    Turn 1-D label array into one-hot encoded matrix 
    """
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())

### DNN layers
def prorate_weight(layer, scale):
    return [w*scale for w in layer.get_weights()]