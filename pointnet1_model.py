import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Dropout, Input)
from pointnet1_layer import TNet, RBFlayer
from common_layer import SMLP, FC


def get_model_pointnet1(
        num_points: int = 1024,
        bn: bool = True, bn_momentum=0.99,
        nkernel: int = 300, kernel: str = None, post_mlps: tuple = None,
        STN: bool = True, STN_Regularization: bool = True,
        keep_prob: float = 0.7, **kwargs):
    print('build model....')
    print('num_points=', num_points)
    print('bn=', bn)
    print('nkernel=', nkernel)
    print('kernel=', kernel)
    print('post_mlps=', post_mlps)
    print('use STN=', STN)
    print('use STN_Regularization=', STN_Regularization)
    print('keep_prob=', keep_prob)

    pc = Input(shape=(num_points, 3), dtype=tf.float32, name='pc_input')  # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    pc_transformed = TNet(
        add_regularization=STN_Regularization, bn_momentum=bn_momentum, use_bias=True, name='pc_transformed')(
        pc) if STN else pc

    if kernel is not None:
        # M: #kernel of rbf
        # RBF extract local feature (B x N x 3) -> (B x M x 1 x N)
        # kernel can be `gau`, `min`
        rbf_out = RBFlayer(nkernel=nkernel, kernel=kernel)(pc_transformed)  # (B, N, M)
        net = tf.transpose(tf.expand_dims(rbf_out, 2), [0, 3, 2, 1])  # (B, M, 1, N)
        if post_mlps:  # (16, 128, 1024)
            # B x M x 1 x N -> B x M x 1 x post_mlps[-1]
            for nfilter in post_mlps:
                net = SMLP(nfilter, 'relu', bn=bn, bn_momentum=bn_momentum)(net)

    else:
        # normal pointnet1 use shared weight mlp
        pc_transformed = tf.expand_dims(pc_transformed, axis=2)  # B x N x 1 x 3
        hidden_64 = SMLP(64, 'relu', bn=bn, bn_momentum=bn_momentum)(pc_transformed)
        embed_64 = SMLP(64, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_64)
        embed_64 = tf.squeeze(embed_64, axis=2)  # BxNx64

        # Feature transformer (B x N x 64 -> B x N x 64)
        feature_transformed = TNet(
            bn_momentum=bn_momentum, add_regularization=True, use_bias=True,
            name='feature_transformed')(embed_64)

        # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
        feature_transformed = tf.expand_dims(feature_transformed, axis=2)
        hidden_64 = SMLP(64, 'relu', bn=bn, bn_momentum=bn_momentum)(feature_transformed)
        hidden_128 = SMLP(128, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_64)
        net = SMLP(1024, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_128)

    # (B x M x 1 x 1024) -> (B x M)
    net = tf.squeeze(net, axis=2)
    global_desc = tf.reduce_max(net, axis=1)

    # FC layers to output k scores (B x M -> B x 40)
    hidden_512 = FC(512, 'relu', bn=bn, bn_momentum=bn_momentum)(global_desc)
    hidden_512 = Dropout(rate=1 - keep_prob)(hidden_512)

    hidden_256 = FC(256, 'relu', bn=bn, bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=1 - keep_prob)(hidden_256)

    logits = FC(40, None, bn=False, name='logits')(hidden_256)

    return keras.Model(inputs=pc, outputs=logits, **kwargs)


if __name__ == '__main__':
    num_points = 1024
    inputs = tf.zeros((32, num_points, 3), dtype=tf.float32)
    model = get_model_pointnet1(
        num_points=num_points,
        kernel='min', nkernel=300, post_mlps=(16, 128, 1024),
        bn_momentum=0.99, bn=True,
        STN=True, STN_Regularization=True,
        keep_prob=0.7
    )
    model.build(inputs.shape)

    outputs = model(inputs, training=True)
    print(f"outputs.shape={outputs.shape}")

    assert outputs.get_shape().as_list() == [32, 40]
