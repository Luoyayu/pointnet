import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Dropout, Dense, BatchNormalization)

from pointnet2_layer import PointNet_SA


class get_pointnet2_model(keras.Model):
    def __init__(self, batch_size: int, bn: bool, bn_momentum, mode, **kwargs):
        super(get_pointnet2_model, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.mode = mode

        self.dense1 = Dense(512, 'relu', name='hidden_512')
        if self.bn: self.bn_fc1 = BatchNormalization()

        self.dropout1 = Dropout(0.3)

        self.dense2 = Dense(128, 'relu', name='hidden_128')
        if self.bn: self.bn_fc2 = BatchNormalization()

        self.dropout2 = Dropout(0.3)

        self.dense3 = Dense(40, 'softmax', name='logits')

    def call(self, point_cloud, training=True):
        print("point_cloud.shape=", point_cloud.shape)
        assert self.mode in ['ssg', 'msg']

        # Hierarchical Point Set Group and Abstract Layer
        num_points = 512 if self.mode == 'ssg' else 1024
        radius = 0.2 if self.mode == 'ssg' else [0.1, 0.2, 0.4]
        samples = 32 if self.mode == 'ssg' else [16, 32, 128]
        filters = [64, 64, 128] if self.mode == 'ssg' else [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        xyz, points = PointNet_SA(
            npoint=num_points, radius=radius, nsample=samples, filters=filters, activation='relu', bn=self.bn,
            bn_momentum=self.bn_momentum, mode=self.mode, group_all=False, name='pointnet_sa_1')(
            point_cloud, None, training=training)

        num_points = 128 if self.mode == 'ssg' else 512
        radius = 0.4 if self.mode == 'ssg' else [0.2, 0.4, 0.8]
        samples = 64 if self.mode == 'ssg' else [32, 64, 128]
        filters = [128, 128, 256] if self.mode == 'ssg' else [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        xyz, points = PointNet_SA(
            npoint=num_points, radius=radius, nsample=samples, filters=filters, activation='relu', bn=self.bn,
            bn_momentum=self.bn_momentum, mode=self.mode, group_all=False, name='pointnet_sa_2')(
            xyz, points, training=training)

        xyz, points = PointNet_SA(
            npoint=None, radius=None, nsample=None, filters=[256, 512, 1024], activation='relu', bn=self.bn,
            bn_momentum=self.bn_momentum, group_all=True, mode='ssg', name='pointnet_sa_3')(
            xyz, points, training=training)

        net = tf.reshape(points, (self.batch_size, -1))

        net = self.dense1(net)
        if self.bn: net = self.bn_fc1(net, training=training)
        net = self.dropout1(net)

        net = self.dense2(net)
        if self.bn: net = self.bn_fc2(net, training=training)
        net = self.dropout2(net)

        logits = self.dense3(net)
        return logits
