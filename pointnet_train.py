import os
import time

import numpy as np
import tensorflow as tf
from pointnet1_model import get_model_pointnet1
from pointnet_dataset import load_hdf5
from pointnet1_config import Config

tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

c = Config()

if c.KERNEL:
    c.BASE_LEARNING_RATE = 0.001
    c.DECAY_STEP = 7000
    c.BN_DECAY_DECAY_STEP = 7000
else:
    c.BASE_LEARNING_RATE = 0.001
    c.DECAY_STEP = 7000
    c.BN_DECAY_DECAY_STEP = 7000

decayed_learning_rate = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=c.BASE_LEARNING_RATE,
    decay_steps=c.DECAY_STEP, decay_rate=c.DECAY_RATE, staircase=True, name='decayed_learning_rate'
)

decayed_bn_momentum = tf.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=c.BN_INIT_DECAY,
    decay_steps=c.BN_DECAY_DECAY_STEP, decay_rate=c.BN_DECAY_DECAY_RATE, staircase=True, name='decayed_bn_momentum'
)


def get_decayed_learning_rate(step: tf.constant):
    return tf.maximum(decayed_learning_rate(step), c.LEARNING_RATE_CLIP)


def get_decayed_bn_momentum(step: tf.constant):
    return tf.minimum(c.BN_DECAY_CLIP, 1 - decayed_bn_momentum(step))


lr = tf.Variable(get_decayed_learning_rate(step=tf.constant(0)), trainable=False)
bn_momentum = tf.Variable(get_decayed_bn_momentum(step=tf.constant(0)), trainable=False)

model = get_model_pointnet1(
    batch_size=c.BATCH_SIZE, num_points=c.NUM_POINT, num_classes=c.NUM_CLASSES,
    activation=c.ACTIVATION, keep_prob=c.KEEP_PROB,

    kernel=c.KERNEL, nkernel=c.NKERNEL, post_mlps=(16, 128, 1024),  # rbf
    bn_momentum=bn_momentum, bn=c.APPLY_BN,
    STN=c.APPLY_STN, STN_Regularization=c.STN_Regularization,

    std_conv=c.with_std_conv,
    name='pointnet1-%s-%d' % (c.KERNEL, c.NKERNEL)
)

if c.USE_WANDB:
    import wandb

    wandb.config.update(c.__dict__)

print(model.summary())
print(c.__dict__)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
classify_loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_one_epoch():
    train_file_idxs = np.arange(0, len(c.TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fidx in range(len(c.TRAIN_FILES)):
        print("<train in %s>" % c.TRAIN_FILES[train_file_idxs[fidx]])
        train_ds = load_hdf5(
            filepath=c.TRAIN_FILES[train_file_idxs[fidx]], num_points=c.NUM_POINT,
            batch_size=c.BATCH_SIZE, training=True, keep_rate=c.INPUT_KEEP_RATE)

        loss_sum = 0
        total_correct = 0
        total_seen = 0
        num_batches = 0

        for feature, label in train_ds:
            train_loss, train_logits = train_step(point_cloud=feature, labels=label)

            pred = tf.math.argmax(train_logits, axis=1)
            label = tf.squeeze(label)
            correct = np.sum(pred == label, dtype=np.float32)

            with c.writer.as_default():
                tf.summary.scalar('train/loss', train_loss, step=optimizer.iterations)
                tf.summary.scalar('train/acc', correct / float(c.BATCH_SIZE), step=optimizer.iterations)
                c.writer.flush()

            total_correct += correct
            total_seen += len(label)
            loss_sum += train_loss
            num_batches += 1
        mean_loss = loss_sum / num_batches
        accuracy = total_correct / total_seen

        with c.writer.as_default():
            tf.summary.scalar("train/mean_loss", mean_loss, step=optimizer.iterations)
            tf.summary.scalar("train/mean_accuracy", accuracy, step=optimizer.iterations)
            c.writer.flush()

        print('train mean loss', mean_loss.numpy())
        print('train accuracy', accuracy)


def eval_one_epoch():
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(c.NUM_CLASSES)]
    total_correct_class = [0 for _ in range(c.NUM_CLASSES)]

    for val_file in c.TEST_FILES:
        val_file = load_hdf5(
            filepath=val_file, num_points=c.NUM_POINT, batch_size=c.BATCH_SIZE, training=False, keep_rate=1.0)

        for feature, label in val_file:
            logits = val_step(point_cloud=feature)

            pred_val = tf.math.argmax(logits, axis=1)
            label = tf.squeeze(label)
            correct = np.sum(pred_val == label)

            total_correct += correct
            total_seen += len(label)

            for i in range(label.shape[0]):
                l = label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += tf.cast(pred_val[i] == l, dtype=tf.int64)

    mean_accuracy = total_correct / float(total_seen)
    avg_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))

    with c.writer.as_default():
        tf.summary.scalar("val/mean_accuracy", mean_accuracy, step=optimizer.iterations)
        tf.summary.scalar("val/avg_class_acc", avg_class_acc, step=optimizer.iterations)
        c.writer.flush()

    print('val mean accuracy', mean_accuracy)
    print('val avg class acc', avg_class_acc)


def train_step(point_cloud, labels):
    with tf.GradientTape() as tape:
        logits = model(point_cloud, training=True)
        loss = tf.reduce_mean(classify_loss_fn(y_pred=logits, y_true=labels)) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss, logits


def val_step(point_cloud):
    logits = model(point_cloud, training=False)
    return logits


def train():
    train_st = time.time()
    for epoch in range(c.MAX_EPOCH):
        epoch_st = time.time()
        print("===== Epoch: %03d =====" % epoch)
        with c.writer.as_default():
            tf.summary.scalar('train/lr', lr, step=epoch)
            tf.summary.scalar('train/bn_momentum', bn_momentum, step=epoch)
            c.writer.flush()

        train_one_epoch()
        lr.assign(get_decayed_learning_rate(step=optimizer.iterations))
        bn_momentum.assign(get_decayed_bn_momentum(step=optimizer.iterations))
        eval_one_epoch()
        model.save_weights("models/model-" + str(epoch) + "-" + str(optimizer.iterations.numpy()), save_format='tf')
        print("cost:[%03d]s // total:[%.2f]m" % (time.time() - epoch_st, (time.time() - train_st) / 60.0))


c.load_dataset()
train()
