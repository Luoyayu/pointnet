import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from common_layer import SMLP, FC


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, use_bias=True, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.use_bias = use_bias
        self.conv0 = SMLP(64, (1, 1), (1, 1), 'relu', bn_momentum=bn_momentum)
        self.conv1 = SMLP(128, (1, 1), (1, 1), 'relu', bn_momentum=bn_momentum)
        self.conv2 = SMLP(1024, (1, 1), (1, 1), 'relu', bn_momentum=bn_momentum)
        self.fc0 = FC(512, activation='relu', bn=True, bn_momentum=bn_momentum)
        self.fc1 = FC(256, activation='relu', bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]
        self.w = self.add_weight(
            shape=(256, self.K * self.K), initializer=tf.zeros_initializer, trainable=True, name='tnet_w')

        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.K, self.K), initializer=tf.zeros_initializer, trainable=True, name='tnet_b')
            self.b = tf.math.add(self.b, tf.constant(np.eye(self.K), dtype=tf.float32))

    def call(self, x, training=None):
        pc = x  # BxNxK

        x = tf.expand_dims(pc, axis=2)  # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)

        x = tf.squeeze(x, axis=2)  # BxNx1024

        # Global features
        # Bx1024
        x = tf.reduce_max(x, axis=1)

        # Fully-connected layers
        x = self.fc0(x, training=training)  # Bx512
        x = self.fc1(x, training=training)  # Bx256

        x = tf.expand_dims(x, axis=1)  # Bx1x256
        # Bx1xK^2
        x = tf.matmul(x, self.w)  # Bx1x256 * 256xK^2 = Bx1xK^2
        x = tf.squeeze(x, axis=1)  # BxK^2
        x = tf.reshape(x, (-1, self.K, self.K))  # BxKxK

        if self.use_bias:
            x += self.b

        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)  # KxK
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(pc, x)


class RBFMin(Layer):
    def __init__(self, output_dim: int):
        super(RBFMin, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.stars = self.add_weight(
            shape=(self.output_dim, input_shape[2]), initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
            trainable=True, name='stars')

    def call(self, inputs, **kwargs):
        diff = inputs[:, None, :, :] - self.stars[None, :, None, :]
        dists = K.min(tf.norm(diff, axis=3), axis=2)
        return dists

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class MyUrsaGau(keras.layers.Layer):
    def __init__(self, output_dim, sigma=0.1, **kwargs):
        self.output_dim = output_dim
        self.sigma = sigma
        self.s = 1 / (2 * sigma * sigma)  # default = 50
        super(MyUrsaGau, self).__init__(**kwargs)

    def build(self, input_shape):
        self.stars = self.add_weight(
            name='stars',
            shape=(self.output_dim, input_shape[2]),
            initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
            trainable=True)
        super(MyUrsaGau, self).build(input_shape)

    def call(self, inputs, **kwargs):
        diff = inputs[:, None, :, :] - self.stars[None, :, None, :]
        dists = K.sum(
            K.exp(-self.s * K.sum(diff * diff, axis=3)),  # Gaussian RBF kernel
            axis=2)  #
        return dists

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class RBFGau(Layer):
    def __init__(self, nkernel=300, **kwargs):
        self.nkernel = nkernel

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.c = self.add_weight(
            name='rbf_s', shape=(self.nkernel, input_shape[-1]),
            initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1),
            trainable=True,
        )

        self.sigma = self.add_weight(
            name="rbf_sigma", shape=(self.nkernel, 1),
            initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1),
            trainable=True,
        )

    def call(self, inputs, training=None):
        assert len(inputs.shape) == 3  # BxNx3
        diff = inputs[:, None, :, :] - self.c[None, :, None, :]
        # (None, 1, None, 3) - (1, 300, 1, 3) = (None, 300, None, 3)
        # tf.reduce_sum(diff ** 2, axis=3).shape  # (None, 300, None)
        g = tf.exp(-1 / (self.sigma ** 2) * tf.reduce_sum(diff ** 2, axis=3))  # g.shape= (None, 300, None)
        return tf.transpose(g, [0, 2, 1])
