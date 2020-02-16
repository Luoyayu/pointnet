import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

CWD = os.getcwd()
DATASET_DIR = os.path.join(CWD, 'data')

writer = tf.summary.create_file_writer("tflogs")

if not os.path.exists(os.path.join(DATASET_DIR, 'modelnet40_ply_hdf5_2048')):
    target_remote = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    target_file = os.path.basename(target_remote)
    target_file_name = target_file.split('.')[0]
    target_file_ext = target_file.split('.')[1]
    assert target_file_ext == 'zip'

    keras.utils.get_file(
        fname=target_file, origin=target_remote,
        extract=True, archive_format=target_file_ext,
        cache_dir=CWD, cache_subdir="data")


# load and sampling num_points from data

def load_hdf5(filepath: str, num_points: int, batch_size: int, training: bool) -> tf.data.Dataset:
    with h5py.File(filepath, mode='r') as f:
        data = f['data'][:, 0:num_points, :]
        data = np.array(data, np.float32)
        label = np.array(f['label'][:], dtype=np.int64)

        ds = tf.data.Dataset.from_tensor_slices((data, label)) \
            .shuffle(2048) \
            .batch(batch_size, drop_remainder=True)
        return ds.map(lambda data_, label_: (data_augment(data_), label_)) \
            .prefetch(tf.data.experimental.AUTOTUNE) if training else ds


def data_augment(batch_data):
    return jitter_point_cloud(rotate_point_cloud(batch_data))


def rotate_point_cloud(batch_data):
    rotated_data = []

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.math.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = tf.constant([[cosval, 0, sinval],
                                       [0, 1, 0],
                                       [-sinval, 0, cosval]], dtype=tf.float32)
        shape_pc = tf.reshape(batch_data[k, ...], (-1, 3))
        rotated_data.append(tf.matmul(shape_pc, rotation_matrix))
    return tf.stack(rotated_data)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    return np.clip(sigma * np.random.randn(*batch_data.shape), -clip, clip) + batch_data
