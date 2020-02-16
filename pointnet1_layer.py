import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

from common_layer import SMLP, FC


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, use_bias=True, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.use_bias = use_bias
        self.conv0 = SMLP(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = SMLP(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = SMLP(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
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
