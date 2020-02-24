import tensorflow as tf
from tensorflow import keras as keras
from common_layer import Conv2d
from pointnet2_utils import sample_and_group_all, sample_and_group

from cpp_shared_modules import (
    gather_point,
    farthest_point_sample,
    query_ball_point,
    group_point
)


class PointNet_SA(keras.layers.Layer):
    pass


class Pointnet_SA(keras.layers.Layer):

    def __init__(
            self, npoint, radius, nsample, mlp, group_all=False, use_xyz=True, activation=tf.nn.relu,
            bn=False
    ):

        super(Pointnet_SA, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.activation = activation
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):

        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(Conv2d(n_filters, activation=self.activation, bn=self.bn))

        super(Pointnet_SA, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        if self.group_all:
            nsample = xyz.get_shape()[1]
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
            )

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points = mlp_layer(new_points, training=training)

        new_points = tf.math.reduce_max(new_points, axis=2, keepdims=True)

        return new_xyz, tf.squeeze(new_points)


class Pointnet_SA_MSG(keras.layers.Layer):

    def __init__(
            self, npoint, radius_list, nsample_list, mlp, use_xyz=True, activation=tf.nn.relu, bn=False
    ):

        super(Pointnet_SA_MSG, self).__init__()

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp = mlp
        self.use_xyz = use_xyz
        self.activation = activation
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):

        for i in range(len(self.radius_list)):
            tmp_list = []
            for i, n_filters in enumerate(self.mlp[i]):
                tmp_list.append(Conv2d(n_filters, activation=self.activation, bn=self.bn))
            self.mlp_list.append(tmp_list)

        super(Pointnet_SA_MSG, self).build(input_shape)

    def call(self, xyz, points, training=True):

        if points is not None:
            if len(points.shape) < 3:
                points = tf.expand_dims(points, axis=0)

        new_xyz = gather_point(xyz, farthest_point_sample(self.npoint, xyz))

        new_points_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])

            if points is not None:
                grouped_points = group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            for i, mlp_layer in enumerate(self.mlp_list[i]):
                grouped_points = mlp_layer(grouped_points, training=training)

            new_points = tf.math.reduce_max(grouped_points, axis=2)
            new_points_list.append(new_points)

        new_points_concat = tf.concat(new_points_list, axis=-1)

        return new_xyz, new_points_concat
