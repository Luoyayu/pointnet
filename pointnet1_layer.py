import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from common_layer import SMLP, FC


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, use_bias=True, activation='relu', **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.use_bias = use_bias
        self.mlp1 = SMLP(64, activation, bn_momentum=bn_momentum)
        self.mlp2 = SMLP(128, activation, bn_momentum=bn_momentum)
        self.mlp3 = SMLP(1024, activation, bn_momentum=bn_momentum)
        self.fc1 = FC(512, activation=activation, bn=True, bn_momentum=bn_momentum)
        self.fc2 = FC(256, activation=activation, bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]
        self.w = self.add_weight(
            shape=(256, self.K * self.K), initializer=tf.zeros_initializer, trainable=True, name='tnet_w')

        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.K, self.K), initializer=tf.zeros_initializer, trainable=True, name='tnet_b')
            self.b = tf.math.add(self.b, tf.constant(np.eye(self.K), dtype=tf.float32))

    def call(self, x, training=None):
        pc = x  # B x N x K

        x = tf.expand_dims(pc, axis=2)  # B x N x 1 x K
        x = self.mlp1(x, training=training)  # B x N x 1 x 64
        x = self.mlp2(x, training=training)  # B x N x 1 x 128
        x = self.mlp3(x, training=training)  # B x N x 1 x 1024

        x = tf.squeeze(x, axis=2)  # B x N x 1024

        # Global features
        # B x 1024
        x = tf.reduce_max(x, axis=1)  # B x N

        # Fully-connected layers
        x = self.fc1(x, training=training)  # B x 512
        x = self.fc2(x, training=training)  # B x 256

        x = tf.expand_dims(x, axis=1)  # B x 1 x 256
        # B x 1 x K^2
        x = tf.matmul(x, self.w)  # Bx1x256 * 256xK^2 = Bx1xK^2
        x = tf.squeeze(x, axis=1)  # B x K^2
        x = tf.reshape(x, (-1, self.K, self.K))  # B x K x K

        if self.use_bias:
            x += self.b

        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)  # K x K
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(pc, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'use_bias': self.use_bias,
        })


class RBFlayer(Layer):
    def __init__(self, nkernel=300, kernel='gau', relation: int = 1, **kwargs):
        self.nkernel = nkernel
        self.kernel = kernel
        self.relation = relation
        assert relation in [1, 2]  # 0: (rbf); 1: (rbf, input);
        assert self.kernel in ['gau', 'min', '']  # 高斯核, 线性核, 无
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.c = self.add_weight(
            name='rbf_c', shape=(self.nkernel, input_shape[-1]),
            initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1),
            trainable=True,
        )  # rbf kernel location (x,y,z)

        if self.kernel == 'gau':
            self.sigma = self.add_weight(
                name="rbf_sigma", shape=(self.nkernel, 1),
                initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1),
                trainable=True,
            )  # rbf kernel width for gua

    def call(self, inputs, training=None):
        # assert len(inputs.shape) == 3  # B x N x 3
        g = inputs
        # (1, 300, 1, 3) - (32, 1, 1024, 3) = (B, M, N, 3)
        diff = self.c[None, :, None, :] - inputs[:, None, :, :]  # rbf kernel active for N points

        if self.kernel == 'gau':
            # tf.reduce_sum(diff ** 2, axis=3).shape  # (32, 300, 1024)
            g = tf.exp(-1 / (self.sigma ** 2) * tf.reduce_sum(diff ** 2, axis=3))  # (B, M, N)
        elif self.kernel == 'min':
            g = K.min(tf.norm(diff, axis=3), axis=2, keepdims=True)

        cg = tf.transpose(g, [0, 2, 1])  # (B, N, M)
        if self.relation == 1:
            # return only rbf
            pass
        elif self.relation == 2:
            # return (rbf, inputs)
            cg = tf.concat([inputs, cg], axis=-1)
        return cg  # (B x N x M)

    def get_config(self):
        config = super(RBFlayer, self).get_config()
        config.update({
            'nkernel': self.nkernel,
            'kernel': self.kernel,
            'relation': self.relation,
        })
