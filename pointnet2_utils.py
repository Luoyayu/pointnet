import tensorflow as tf

from cpp_shared_modules import (
    farthest_point_sample,
    gather_point, query_ball_point, group_point, knn_point
)


def sample_and_group_all(xyz: tf.Tensor, points: tf.Tensor, use_xyz: bool = True):
    # xyz: BxNx3
    batch_size, num_points, _ = xyz.get_shape().as_list()
    new_xyz = tf.zeros(shape=(batch_size, 1, 3))  # Bx1x3
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, num_points, 3))  # Bx1xNx3
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (BxNx(C+3))
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, axis=1)  # Bx1xNx(C+3)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group(num_points, radius, samples, xyz, points, use_xyz=True):
    new_xyz = gather_point(xyz, farthest_point_sample(num_points, xyz))  # BxNx3
    idx, pts_cnt = query_ball_point(radius, samples, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (BxNxSx3)
    if points is not None:
        grouped_points = group_point(points, idx)  # (BxNxSx3)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (BxNxSx(C+3))
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points
