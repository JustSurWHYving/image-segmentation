# imports
import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os.path import join, isdir
from os import listdir, rmdir
from shutil import move, rmtree, make_archive
from PIL import Image
from keras.src.models.model import Model
from keras.src.layers import *

# directories
DRIVE_DIR = 'drive/'
CONTENT_DIR = 'content/'
GT_DIR = CONTENT_DIR + 'gtFine/'
IMG_DIR = CONTENT_DIR + 'leftImg8bit/'


# collapse child directories
for parent in listdir(GT_DIR):
    parent_dir = GT_DIR + parent
    for child in listdir(parent_dir):
        if isdir(join(parent_dir, child)):
            keep = glob.glob(join(parent_dir, child) + '/*_gtFine_color.png')
            keep = [f.split('/')[-1] for f in keep]
            for filename in list(set(listdir(join(parent_dir, child))) & set(keep)):
                move(join(parent_dir, child, filename), join(parent_dir, filename))
            rmtree(join(parent_dir, child))

for parent in listdir(IMG_DIR):
    parent_dir = IMG_DIR + parent
    for child in listdir(parent_dir):
        if isdir(join(parent_dir, child)):
            for filename in listdir(join(parent_dir, child)):
                move(join(parent_dir, child, filename), join(parent_dir, filename))
            rmtree(join(parent_dir, child))


# normalize image pixels (z-score to impliment)
IMG_SIZE = 299
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE  # auto tunes the pipeline's performance

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img /= 255.0
    return img


def get_image_paths(dir):
    return sorted([dir + path for path in listdir(dir)])


# create tf.Dataset objects
gt_train_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'train/'))
gt_val_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'val/'))
gt_test_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(GT_DIR + 'test/'))

gt_train_ds = gt_train_ds.map(load_and_preprocess_image)
gt_val_ds = gt_val_ds.map(load_and_preprocess_image)
gt_test_ds = gt_test_ds.map(load_and_preprocess_image)

im_train_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'train/'))
im_val_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'val/'))
im_test_ds = tf.data.Dataset.from_tensor_slices(get_image_paths(IMG_DIR + 'test/'))

im_train_ds = im_train_ds.map(load_and_preprocess_image)
im_val_ds = im_val_ds.map(load_and_preprocess_image)
im_test_ds = im_test_ds.map(load_and_preprocess_image)


# pspnet architecture implimentation
def conv_block(X, filters, block):
    # resiudal block with dilated convolutions
    # add skip connection at last after doing convoluion

    b = 'block_' + str(block) + '_'
    f1, f2, f3 = filters
    X_skip = X

    # block_a
    X = Conv2D(filters=f1, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'a')(X)
    X = BatchNormalization(name=b + 'batch_norm_a')(X)
    X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_a')(X)
    # block_b
    X = Conv2D(filters=f2, kernel_size=(3, 3), dilation_rate=(2, 2),
               padding='same', kernel_initializer='he_normal', name=b + 'b')(X)
    X = BatchNormalization(name=b + 'batch_norm_b')(X)
    X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_b')(X)
    # block_c
    X = Conv2D(filters=f3, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'c')(X)
    X = BatchNormalization(name=b + 'batch_norm_c')(X)
    # skip_conv
    X_skip = Conv2D(filters=f3, kernel_size=(3, 3), padding='same', name=b + 'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b + 'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b + 'add')([X, X_skip])
    X = ReLU(name=b + 'relu')(X)
    return X


def base_feature_maps(input_layer):
    # base covolution module to get input image feature maps

    # block_1
    base = conv_block(input_layer, [16, 16, 32], '1')
    # block_2
    base = conv_block(base, [16, 16, 32], '2')
    return base


def pyramid_feature_maps(input_layer):
    # pyramid pooling module

    base = base_feature_maps(input_layer)
    # red
    red = GlobalAveragePooling2D(name='red_pool')(base)
    red = tf.keras.layers.Reshape((1, 1, 32))(red)
    red = Conv2D(filters=32, kernel_size=(1, 1), name='red_1_by_1')(red)
    red = UpSampling2D(size=128, interpolation='bilinear', name='red_upsampling')(red)
    red = tf.image.resize(red, [IMG_SIZE, IMG_SIZE])
    # yellow
    yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
    yellow = Conv2D(filters=32, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
    yellow = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow)
    yellow = tf.image.resize(yellow, [IMG_SIZE, IMG_SIZE])
    # blue
    blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base)
    blue = Conv2D(filters=32, kernel_size=(1, 1), name='blue_1_by_1')(blue)
    blue = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue)
    blue = tf.image.resize(blue, [IMG_SIZE, IMG_SIZE])
    # green
    green = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base)
    green = Conv2D(filters=32, kernel_size=(1, 1), name='green_1_by_1')(green)
    green = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green)
    green = tf.image.resize(green, [IMG_SIZE, IMG_SIZE])
    # base + red + yellow + blue + green
    return tf.keras.layers.concatenate([base, red, yellow, blue, green])


def last_conv_module(input_layer):
    X = pyramid_feature_maps(input_layer)
    X = Conv2D(filters=3, kernel_size=3, padding='same', name='last_conv_3_by_3')(X)
    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
    X = Activation('sigmoid', name='last_conv_relu')(X)
    return X

# input and output layer dims
input_shape = list(im_train_ds.take((1)))[0].shape
input_layer = Input(shape=input_shape, name='input')
output_layer = last_conv_module(input_layer)

model = Model(inputs=input_layer, outputs=output_layer)
print(model.summary)