import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Dropout, Input)
from pointnet1_layer import TNet, RBFlayer
from common_layer import SMLP, FC


def get_model_pointnet1(
        num_points: int = 1024, batch_size=32, num_classes=40, activation='relu',
        bn: bool = True, bn_momentum=0.99,
        nkernel: int = 300, kernel: str = None, post_mlps: tuple = None,
        STN: bool = True, STN_Regularization: bool = True,
        keep_prob: float = 0.7, std_conv=False, relation=1, **kwargs):
    print('build model....')
    print('batch_size=', batch_size)
    print('num_points=', num_points)
    print('num_classes=', num_classes)
    print('bn=', bn)
    print('#kernel=', nkernel)
    print('rbf kernel=', kernel)
    print('use post mlps after rbf=', post_mlps)
    print('use STN=', STN)
    print('use STN Regularization=', STN_Regularization)
    print('keep_prob=', keep_prob)
    print('use std conv=', std_conv)

    pc = Input(shape=(None, 3), dtype=tf.float32, name='pc_input')  # BxNx3

    # Input transformer (BxNx3 -> BxNx3)
    pc_transformed = TNet(
        add_regularization=STN_Regularization, bn_momentum=bn_momentum,
        use_bias=True, name='pc_transformed')(pc) if STN else pc

    if kernel is not None:
        # M: #kernel of rbf
        # RBF extract local feature (BxNx3) -> (BxMx1xN)
        # kernel can be `gau`, `min`
        rbf_out = RBFlayer(nkernel=nkernel, kernel=kernel, relation=relation)(pc_transformed)  # BxNxM

        if relation == 1:  # high level pc
            h1 = SMLP(64, activation, bn=bn, bn_momentum=bn_momentum)(tf.expand_dims(pc_transformed, 2))
            h2 = SMLP(64, activation, bn=bn, bn_momentum=bn_momentum)(h1)
            h3 = SMLP(nkernel, activation, bn=bn, bn_momentum=bn_momentum)(h2)  # BxNx1xM
            rbf_out = tf.math.multiply(rbf_out, tf.squeeze(h3, 2))
        net = tf.expand_dims(rbf_out, 2)  # BxNx1xM
        if post_mlps:  # (16, 128, 1024)
            # B x N x 1 x M -> B x N x 1 x post_mlps[-1]
            for nfilter in post_mlps:
                net = SMLP(nfilter, activation, bn=bn, bn_momentum=bn_momentum)(net)

    else:
        # normal pointnet1 use shared mlp
        pc_transformed = tf.expand_dims(pc_transformed, axis=2)  # BxNx1x3
        hidden_64 = SMLP(64, activation, bn=bn, bn_momentum=bn_momentum)(pc_transformed)
        embed_64 = SMLP(64, activation, bn=bn, bn_momentum=bn_momentum)(hidden_64)
        embed_64 = tf.squeeze(embed_64, axis=2)  # BxNx64

        # Feature transformer (BxNx64 -> BxNx64)
        feature_transformed = TNet(
            bn_momentum=bn_momentum, add_regularization=True, use_bias=True,
            name='feature_transformed')(embed_64)  # BxNx64

        # Embed to 1024-dim space (BxNx64 -> BxNx1x1024)
        feature_transformed = tf.expand_dims(feature_transformed, axis=2)  # BxNx1x64
        hidden_64 = SMLP(64, activation, bn=bn, bn_momentum=bn_momentum)(feature_transformed)  # BxNx1x64
        hidden_128 = SMLP(128, activation, bn=bn, bn_momentum=bn_momentum)(hidden_64)  # BxNx1x128
        net = SMLP(1024, activation, bn=bn, bn_momentum=bn_momentum)(hidden_128)  # BxNx1x1024

    # aggregation
    # (B x N x 1 x 1024) -> (B x N) pointnet1
    # (B x N x 1 x M) -> (B x M) pointnet1 wo post mlp
    # (B x N x 1 x post_mlp[-1]) -> (B x M) pointnet1 with post mlp
    net = tf.squeeze(net, axis=2)  # B x N x 1024
    global_desc = tf.reduce_max(net, axis=1)  # B x 1024

    # FC layers to output k scores (B x M -> B x 40) or (B x 1024 -> B x 40)
    hidden_512 = FC(512, activation, bn=bn, bn_momentum=bn_momentum)(global_desc)
    hidden_512 = Dropout(rate=1 - keep_prob)(hidden_512)

    hidden_256 = FC(256, activation, bn=bn, bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=1 - keep_prob)(hidden_256)

    logits = FC(num_classes, None, bn=False, name='logits')(hidden_256)

    return keras.Model(inputs=pc, outputs=logits, **kwargs)


if __name__ == '__main__':
    num_points = 1024
    inputs = tf.zeros((32, num_points, 3), dtype=tf.float32)
    model = get_model_pointnet1(
        batch_size=32, num_points=1024, num_classes=40,
        activation='relu', keep_prob=0.7,

        kernel='gau', nkernel=300, post_mlps=(16, 128, 1024),  # rbf
        bn_momentum=0.99, bn=True, relation=1,
        STN=True, STN_Regularization=True,

        std_conv=False,
        name='pointnet1-%s-%d' % ('gau', 300)
    )
    print(model.summary())
    keras.utils.plot_model(model, 'model.png', show_shapes=True)
    # model.build(inputs.shape)

    outputs = model(inputs, training=True)
    print(f"outputs.shape={outputs.shape}")

    assert outputs.get_shape().as_list() == [32, 40]
