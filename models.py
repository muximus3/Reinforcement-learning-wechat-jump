# -*- coding: utf-8 -*-
# @Time    : 2018/2/28 下午6:03
# @Author  : Eric
# @Site    : 
# @File    : models.py
# @Software: PyCharm
import os
import sys

_root = os.path.normpath("%s/../.." % os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
import numpy as np
from keras.layers import *
from keras import Sequential, models
from keras.models import Model
from keras.optimizers import *
from keras.initializers import *
from PIL import Image
from keras.losses import *
from keras.datasets import mnist
from keras.utils import to_categorical


class LR(object):
    def build_model(self):
        model = Sequential()
        model.add(Dense(1, use_bias=True, input_shape=(1,)))
        model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
        return model


class CNNModel(object):
    def __init__(self, filters, num_filters, input_shape, action_space):
        self.filters = filters
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.action_space = action_space

    def build_model(self):
        inputs = Input(self.input_shape)
        conv_blocks = []
        for i, fz in enumerate(self.filters):
            conv2d = Conv2D(filters=self.num_filters[i], kernel_size=fz, strides=2, padding='same', activation='relu', data_format='channels_last')(inputs)
            conv2d = MaxPool2D()(conv2d)
            conv2d = Flatten()(conv2d)
            conv_blocks.append(conv2d)
        convs = concatenate(conv_blocks, axis=-1) if len(conv_blocks) > 1 else conv_blocks[0]
        out = Dense(500, activation='relu')(convs)
        out = Dense(self.action_space)(out)
        model = Model(inputs, out)
        model.compile(optimizer='adam', loss=mse)
        return model


class MNIST(object):
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(25, (5, 5), padding='same', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(2, 2))
        model.add(Activation('relu'))
        model.add(Convolution2D(50, (5, 5), padding='same'))
        model.add(MaxPooling2D(2, 2))
        model.add(Activation('relu'))
        model.add(Convolution2D(40, (4, 4), padding='same'))
        model.add(MaxPooling2D(2, 2))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', kernel_initializer=RandomUniform(-0.1, 0.1, seed=1400),
                        activity_regularizer=regularizers.l2(1e-04)))
        # compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def pic2matrix(img_dir):
    files = os.listdir(img_dir)
    images = np.zeros((len(files), 28, 28, 1))
    for i, image in enumerate(files):
        im = Image.open(os.path.join(img_dir, image))
        im = im.convert("L")
        data = im.getdata()
        data = np.asarray(data).reshape((1, 28, 28, 1))
        images[i] = data
    return images


def upsample_total(from_num, to_num):
    assert to_num > from_num
    left_num = to_num - from_num
    sample_indice = np.random.randint(0, from_num, left_num, dtype=np.int)
    return sample_indice


def train_mnist():
    model = MNIST().build_model()
    new_images = pic2matrix('images_train')
    new_images_y = to_categorical(np.array([3, 6, 2, 4, 0, 8, 5, 1, 7, 9]), 10)
    indice = upsample_total(len(new_images), 10000)
    new_images = new_images[indice]
    new_images_y = new_images_y[indice]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)
    x_train = np.concatenate((x_train, new_images, x_test))
    y_train = np.concatenate((y_train, new_images_y, y_test))
    print(x_train.shape)
    print(y_train.shape)

    model.fit(x_train, y_train, validation_split=0.01, batch_size=50, epochs=6, verbose=2)
    # print(model.evaluate(x_test, y_test, batch_size=100))
    model.save('mnist2.h5')


if __name__ == '__main__':
    # model = CNNModel((8, 4, 3), (100, 80, 80), (80, 80, 1,), 15).build_model()
    from keras.utils import to_categorical

    model = LR().build_model()
    x_test = np.linspace(0, 4, 9999)
    np.random.shuffle(x_test)
    y_test = 0.5 * x_test + np.random.normal(0, 0.05, (9999,))
    # y_test = to_categorical(y_test * 10, 22)
    print(x_test.shape, y_test.shape)
    model.fit(x_test, y_test, epochs=2)
    a = model.predict(np.asarray([1, 2, 3, 4]))
    print(a)
