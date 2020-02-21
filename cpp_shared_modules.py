import tensorflow as tf
from tensorflow.python.framework import ops

sampling_module = tf.load_op_library('pointnet/tf_ops/tf_sampling_so.so')
grouping_module = tf.load_op_library('pointnet/tf_ops/tf_grouping_so.so')
interpolate_module = tf.load_op_library('pointnet/tf_ops/tf_interpolate_so.so')


def prob_sample(inp, inpr):
    """
    :param inp:     B * ncategory
    :param inpr:    B * N
    :return:        B * N
    """
    return sampling_module.prob_sample(inp, inpr)


ops.NoGradient('ProbSample')


def gather_point(inp, idx):
    # 调整 并行 后的中心点 idx -> new_xyz
    """
    :param inp:     B * N * 3
    :param idx:     B * N
    :return:        B * N * 3
    """
    return sampling_module.gather_point(inp, idx)


@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op, out_g):
    inp = op.inputs[0]
    idx = op.inputs[1]
    return [sampling_module.gather_point_grad(inp, idx, out_g), None]


def farthest_point_sample(npoint, inp):
    """
    :param npoint:  int32
    :param inp:     B * ndataset * 3
    :return:        B * N
    """
    return sampling_module.farthest_point_sample(inp, npoint)


ops.NoGradient('FarthestPointSample')


def query_ball_point(radius, nsample, xyz1, xyz2):
    """
    :param radius: ball search radius
    :param nsample: S number of points selected in each ball region
    :param xyz1: input points (BxAx3) A: all points
    :param xyz2: query points (BxNx3) N:
    :return:
        idx:        B * N * S       indices to input points
        pts_cnt:    B * N           number of unique points
    """
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)


ops.NoGradient('QueryBallPoint')


def select_top_k(k, dist):
    """
    :param k: number of k SMALLEST elements selected
    :param dist: distance matrix, m query points, n dataset points
    :return: 
        idx:        b * m * n   first k in n are indices to the top k
        dist_out:   b * m * n   first k in n are the top k
    """
    return grouping_module.selection_sort(dist, k)


ops.NoGradient('SelectionSort')


def group_point(points, idx):
    """
    :param points:  B * ndataset * channel    points to sample from
    :param idx:     B * N * nsample           indices to points
    :return: out:   B * N * nsample * channel  values sampled from points
    """
    return grouping_module.group_point(points, idx)


@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]


def knn_point(k, xyz1, xyz2):
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print(b, n, c, m)
    print(xyz1, (b, 1, n, c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)
    print(dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])
    print(idx, val)
    # val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx


def three_nn(xyz1, xyz2):
    """
    :param xyz1: b * n * 3  unknown points
    :param xyz2: b * m * 3  known points
    :return:
        dist: b * n * 3  distances to known points
        idx:  b * n * 3  indices to known points
    """
    return interpolate_module.three_nn(xyz1, xyz2)


ops.NoGradient('ThreeNN')


def three_interpolate(points, idx, weight):
    """
    :param points:  b * m * c   known points
    :param idx:     b * n * 3   indices to known points
    :param weight:  b * n * 3   weights on known points
    :return:  out:  b * n * c   interpolated point values
    """
    return interpolate_module.three_interpolate(points, idx, weight)


@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]
