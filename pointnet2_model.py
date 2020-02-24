import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Input, Dropout)
from pointnet2_layer import PointNet_SA
from common_layer import FC


def get_pointnet2_model(bn: bool, bn_momentum, mode, **kwargs):
    assert mode in ['ssg', 'msg']
    point_cloud = Input(shape=(None, 3), dtype=tf.float32, name='pt_cloud_input')  # BxNx3

    # Hierarchical Point Set Group and Abstract Layer
    num_points = 512 if mode == 'ssg' else 1024
    radius = 0.2 if mode == 'ssg' else [0.1, 0.2, 0.4]
    samples = 32 if mode == 'ssg' else [16, 32, 128]
    filters = [64, 64, 128] if mode == 'ssg' else [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
    xyz, points = PointNet_SA(
        npoint=num_points, radius=radius, nsample=samples, filters=filters, activation='relu', bn=bn,
        bn_momentum=bn_momentum, mode=mode, group_all=False, name='pointnet_sa_1')(point_cloud, None)

    num_points = 128 if mode == 'ssg' else 512
    radius = 0.4 if mode == 'ssg' else [0.2, 0.4, 0.8]
    samples = 64 if mode == 'ssg' else [32, 64, 128]
    filters = [128, 128, 256] if mode == 'ssg' else [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
    xyz, points = PointNet_SA(
        npoint=num_points, radius=radius, nsample=samples, filters=filters, activation='relu', bn=bn,
        bn_momentum=bn_momentum, mode=mode, group_all=False, name='pointnet_sa_2')(xyz, points)

    xyz, points = PointNet_SA(
        npoint=None, radius=None, nsample=None, filters=[256, 512, 1024], activation='relu', bn=bn,
        bn_momentum=bn_momentum, group_all=True, mode='ssg', name='pointnet_sa_3')(xyz, points)

    x = tf.reshape(points, (points.get_shape()[0], -1))

    hidden_512 = Dropout(FC(512, activation='relu', bn=bn, bn_momentum=bn_momentum, name='hidden_512')(x))
    hidden_128 = Dropout(FC(128, activation='relu', bn=bn, bn_momentum=bn_momentum, name='hidden_128')(hidden_512))
    logits = Dropout(FC(40, activation='softmax', bn=bn, bn_momentum=bn_momentum, name='output_logits')(hidden_128))

    return keras.Model(inputs=point_cloud, outputs=logits, **kwargs)
