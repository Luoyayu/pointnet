from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, Dense
)


# shared weight MLP implement by Conv2D
class SMLP(Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None,
                 bn=True, bn_momentum=0.99, initializer='glorot_normal', **kwargs):
        super(SMLP, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, activation=activation, use_bias=not bn,
                           kernel_initializer=initializer)
        if bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        assert training is not None
        x = self.conv(x)
        if self.bn:
            x = self.bn(x, training=training)
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