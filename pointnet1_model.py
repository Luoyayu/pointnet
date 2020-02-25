import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Dropout, Input)
from pointnet1_layer import TNet, RBFMin, RBFGau
from common_layer import SMLP, FC


def get_pointnet1_model(bn_momentum, bn: bool = True, **kwargs):
    pc = Input(shape=(None, 3), dtype=tf.float32, name='point_cloud_input')  # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    point_cloud_transformed = TNet(bn_momentum=bn_momentum, use_bias=True, name='point_cloud_transformed')(pc)

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, axis=2)  # BxNx1x3 for weight shared MLP

    hidden_64 = SMLP(64, (1, 1), (1, 1), 'relu', bn=bn, bn_momentum=bn_momentum)(point_cloud_transformed)
    embed_64 = SMLP(64, (1, 1), (1, 1), 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_64)
    embed_64 = tf.squeeze(embed_64, axis=2)  # BxNx64

    # Feature transformer (B x N x 64 -> B x N x 64)
    feature_transformed = TNet(bn_momentum=bn_momentum, add_regularization=True, use_bias=True,
                               name='feature_transformed')(embed_64)

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    feature_transformed = tf.expand_dims(feature_transformed, axis=2)
    hidden_64 = SMLP(64, (1, 1), (1, 1), 'relu', bn=bn, bn_momentum=bn_momentum)(feature_transformed)
    hidden_128 = SMLP(128, (1, 1), (1, 1), 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_64)
    embed_1024 = SMLP(1024, (1, 1), (1, 1), 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_128)

    embed_1024 = tf.squeeze(embed_1024, axis=2)

    # Aggregate whole point to global feature (B x N x 1024 -> B x 1024)
    global_shape_desc = tf.reduce_max(embed_1024, axis=1)

    # FC layers to output k scores (B x 1024 -> B x 40)
    hidden_512 = FC(512, 'relu', bn=bn, bn_momentum=bn_momentum)(global_shape_desc)
    hidden_512 = Dropout(rate=0.3)(hidden_512)

    hidden_256 = FC(256, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=0.3)(hidden_256)

    logits = FC(40, None, bn=False, name='logits')(hidden_256)

    return keras.Model(inputs=pc, outputs=logits, **kwargs)
    # return keras.Model(inputs=pc, outputs=[logits, point_cloud_transformed], **kwargs)


def get_pointnet1_ursa_model(bn_momentum, bn: bool = True, **kwargs):
    pc = Input(shape=(None, 3), dtype=tf.float32, name='point_cloud_input')  # BxNx3

    x = RBFMin(output_dim=256)(pc)
    x = keras.activations.relu(x)
    x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    x = FC(512, activation='relu', bn=bn, bn_momentum=bn_momentum, name='fc_512')(x)
    x = FC(256, activation='relu', bn=bn, bn_momentum=bn_momentum, name='fc_256')(x)
    x = Dropout(rate=0.3)(x)
    logits = FC(40, 'softmax')(x)

    return keras.Model(inputs=pc, outputs=logits, **kwargs)


def get_pointnet1_rbf(bn_momentum, bn: bool = True, nkernel=300, kernel='gau', **kwargs):
    pc = Input(shape=(None, 3), dtype=tf.float32, name='point_cloud_input')  # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    point_cloud_transformed = TNet(bn_momentum=bn_momentum, use_bias=True, name='point_cloud_transformed')(pc)

    rbf_out = RBFGau(nkernel)(point_cloud_transformed)

    # Aggregate whole point to global feature (B x N x nkernel -> B x N)
    global_shape_desc = tf.reduce_max(rbf_out, axis=1)

    # FC layers to output k scores (B x N -> B x 40)
    hidden_512 = FC(512, 'relu', bn=bn, bn_momentum=bn_momentum)(global_shape_desc)
    hidden_512 = Dropout(rate=0.3)(hidden_512)

    hidden_256 = FC(256, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=0.3)(hidden_256)

    logits = FC(40, None, bn=False, name='logits')(hidden_256)

    return keras.Model(inputs=pc, outputs=logits, **kwargs)


if __name__ == '__main__':
    inputs = tf.zeros((32, 1024, 3), dtype=tf.float32)
    model = get_pointnet1_rbf(nkernel=300, kernel='gau', bn_momentum=0.99, bn=True)
    model.build(inputs.shape)
    outputs = model(inputs, training=True)

    print(f"outputs.shape={outputs.shape}")

    assert outputs.get_shape().as_list() == [32, 40]
