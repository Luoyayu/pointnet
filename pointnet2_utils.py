import tensorflow as tf

from cpp_shared_modules import (
    farthest_point_sample,
    gather_point, query_ball_point, group_point, knn_point
)


def sample_and_group_all(xyz: tf.Tensor, points: tf.Tensor, use_xyz: bool = True):
    """

    :param xyz:     BxNx3
    :param points:  BxNxC abstracted local features
    :param use_xyz: if True concat XYZ with local point features
    :return:
        new_xyz: Bx1x3
        new_points: Bx1xNx(C+3)
    """
    batch_size, num_points, _ = xyz.get_shape().as_list()
    new_xyz = tf.zeros(shape=(batch_size, 1, 3), dtype=tf.float32)  # Bx1x3 (0,0,0)作为质心
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
    """

    :param num_points: 采样 N 个质点
    :param radius: 采样半径
    :param samples: 采样半径内聚集 S 个点
    :param xyz: BxNx3 输入坐标
    :param points: BxNxC 输入特征
    :param use_xyz:
    :return: 中心点坐标和聚组后的点(C+3)
    """
    new_xyz = gather_point(xyz, farthest_point_sample(num_points, xyz))  # BxNx3 # 采样后的中心点坐标 (BxNx3)
    idx, _ = query_ball_point(radius, samples, xyz, new_xyz)  # 查询以new_xyz为中心, radius为半径内的组内的点idx (BxNxS)
    grouped_xyz = group_point(xyz, idx)  # (BxNxSx3) # 聚组
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, samples, 1])  # 中心化 (BxNxSx3)

    if points is not None:
        grouped_points = group_point(points, idx)  # 对高维空间也做聚组 (BxNxSxC)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (BxNxSx(C+3))
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz  # 只有点的坐标输入

    return new_xyz, new_points
