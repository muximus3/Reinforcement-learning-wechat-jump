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
            conv2d = Dropout(0.5)(conv2d)
            conv2d = GlobalMaxPool2D(data_format='channels_last')(conv2d)
            conv_blocks.append(conv2d)
        convs = concatenate(conv_blocks, axis=-1) if len(conv_blocks) > 1 else conv_blocks[0]
        out = Dense(500, activation='relu')(convs)
        out = Dense(self.action_space)(out)
        model = Model(inputs, out)
        model.compile(optimizer='adam', loss='mse')
        return model


if __name__ == '__main__':
    model = CNNModel((8, 4, 3), (100, 80, 80), (80, 80,1), 15).build_model()
    a  = model.predict(np.array([np.zeros((80,80,1))]))
    print(a)
