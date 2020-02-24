import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, Dense
)

BatchNormalization._USE_V2_BEHAVIOR = False


# shared weight MLP implement by Conv2D
class SMLP(Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), activation=None, padding='valid',
                 bn=True, bn_momentum=0.99, initializer='glorot_normal', **kwargs):
        super(SMLP, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, activation=activation, use_bias=not self.bn,
                           kernel_initializer=initializer)
        if type(activation) == str:
            self.activation = keras.activations.get(activation)
        if bn:
            self.bn_fn = BatchNormalization(momentum=bn_momentum, fused=False)

    @tf.function
    def call(self, x, training=None):
        assert training is not None
        x = self.conv(x)
        if self.bn:
            x = self.bn_fn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(Layer):

    def __init__(self, filters, strides=(1, 1), activation=tf.nn.relu, padding='VALID', initializer='glorot_normal',
                 bn=False, bn_momentum=0.99):
        super(Conv2d, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn
        self.bn_momentum = bn_momentum
        if type(activation) == str:
            self.activation = keras.activations.get(activation)
        if bn:
            self.bn_fn = BatchNormalization(momentum=bn_momentum, fused=False)

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name='pnet_conv'
        )

        if self.bn: self.bn_layer = BatchNormalization(momentum=self.bn_momentum)

        super(Conv2d, self).build(input_shape)

    def call(self, inputs, training=True):

        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn: points = self.bn_layer(points, training=training)

        if self.activation: points = self.activation(points)

        return points


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

    @tf.function
    def call(self, x, training=None):
        assert training is not None
        x = self.dense(x)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x
