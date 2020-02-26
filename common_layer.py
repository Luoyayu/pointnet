import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, Dense
)


# shared weight MLP implement by Conv2D
class SMLP(Layer):
    def __init__(
            self, filters, activation=None, kernel_size=(1, 1), strides=(1, 1),
            padding='valid', bn=True, bn_momentum=0.99,
            initializer='glorot_normal', **kwargs):
        super(SMLP, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(
            filters, kernel_size, strides,
            padding=padding, activation=activation, use_bias=not self.bn,
            kernel_initializer=initializer)
        if type(activation) == str:
            self.activation = keras.activations.get(activation)
        if bn:
            self.bn_fn = BatchNormalization(momentum=bn_momentum)

    def call(self, x, training=None):
        assert training is not None
        x = self.conv(x)
        if self.bn:
            x = self.bn_fn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FC(Layer):
    def __init__(self, units, activation=None, bn=True, bn_momentum=0.99, **kwargs):
        super(FC, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not bn)
        if type(activation) == str:
            self.activation = keras.activations.get(activation)
        if bn:
            self.bn = BatchNormalization(momentum=bn_momentum)

    def call(self, x, training=None):
        assert training is not None
        x = self.dense(x)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x
