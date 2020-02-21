import tensorflow as tf
from tensorflow.keras.layers import Layer
from common_layer import SMLP
from pointnet2_utils import sample_and_group_all, sample_and_group

from cpp_shared_modules import (
    gather_point,
    farthest_point_sample,
    query_ball_point,
    group_point
)


# single-scale grouping, SSG
# multi-scale grouping,  MSG

class PointNet_SA(Layer):
    def __init__(self, num_points, radius, samples, filters: list, use_xyz: bool = True,
                 activation='relu', bn=True, bn_momentum=0.99, mode: str = 'basic', group_all=False,
                 initializer='glorot_normal', **kwargs):
        super().__init__(**kwargs)
        self.num_point = num_points
        self.radius = radius
        self.samples = samples
        self.filters = filters  # filters depth list
        self.use_xyz = use_xyz
        self.activation = activation
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.mode = mode
        self.mlps = []  # smlp list [[], [], []] or []
        self.group_all = group_all
        self.initializer = initializer
        if mode not in ['ssg', 'msg']:
            NotImplementedError("mode must 'ssg' or 'msg'")

    def build(self, input_shape):
        if self.mode == 'ssg':
            for i, filter in enumerate(self.filters):
                self.mlps.append(
                    SMLP(filter, activation=self.activation, bn=self.bn, bn_momentum=self.bn_momentum,
                         initializer=self.initializer))
        else:
            for i, _ in enumerate(self.radius):
                mlps = []
                for filter in self.filters[i]:
                    mlps.append(SMLP(filter, activation=self.activation, bn=self.bn, bn_momentum=self.bn_momentum,
                                     initializer=self.initializer))
                self.mlps.append(mlps)
        super(PointNet_SA, self).build(input_shape)

    @tf.function
    def call(self, inputs, xyz, training=None, **kwargs):
        if self.mode == 'ssg':
            new_xyz, new_points = \
                sample_and_group_all(xyz, inputs, self.use_xyz) if self.group_all else \
                    sample_and_group(self.num_point, self.radius, self.samples, xyz, inputs, self.use_xyz)
            print('new_xyz.shape=', new_xyz.shape)
            print('new_points.shape=', new_points.shape)

            for mlp in self.mlps:
                new_points = mlp(new_points, training=training)
            new_points = tf.math.reduce_max(new_points, 2, keepdims=True)
            return new_xyz, tf.squeeze(new_points)
        elif self.mode == 'msg':
            new_xyz = gather_point(xyz, farthest_point_sample(self.num_point, xyz))
            new_points_list = []

            for i, r in enumerate(self.radius):
                sample = self.samples[i]
                idx, pts_cnt = query_ball_point(r, sample, xyz, new_xyz)
                grouped_xyz = group_point(xyz, idx)
                grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, sample, 1])
                if inputs is not None:
                    grouped_points = group_point(inputs, idx)
                    if self.use_xyz:
                        grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
                else:
                    grouped_points = grouped_xyz

                for mlp in self.mlps:
                    grouped_points = mlp(grouped_points, training=training)

                new_points = tf.math.reduce_max(grouped_points, axis=2)
                new_points_list.append(new_points)

            new_points_concat = tf.concat(new_points_list, axis=-1)

            return new_xyz, new_points_concat
        elif self.mode == 'mrg':
            pass
