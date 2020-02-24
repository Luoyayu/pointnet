import os
import sys
import numpy as np
import tensorflow as tf
from pointnet1_model import get_pointnet1_model, get_pointnet1_ursa_model
from pointnet2_model import get_pointnet2_model
from poinrnet_dataset import load_hdf5, writer

tf.random.set_seed(0)


class Config:
    def __init__(self):
        self.MAX_EPOCH: int = 250
        self.BATCH_SIZE = 32
        self.NUM_CLASSES = 40
        self.NUM_POINT = 1024
        self.MAX_NUM_POINT = 2048

        self.BASE_LEARNING_RATE = 0.001
        self.LEARNING_RATE_CLIP = 1e-5
        self.MOMENTUM = 0.99

        self.OPTIMIZER: str = 'adam'

        self.DECAY_STEP = 7000
        self.DECAY_RATE = 0.7

        self.APPLY_BN = True
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_CLIP = 0.99
        self.BN_DECAY_DECAY_STEP = 7000

        self.TRAIN_FILES: list = []
        self.TEST_FILES: list = []

        self.USE_V2 = True  # pointnet++
        self.USE_WANDB = False

    def load_dataset(self):
        train_files_path = os.path.join('data', 'modelnet40_ply_hdf5_2048', 'train_files.txt')
        if not os.path.exists(train_files_path):
            sys.exit(f'not found {train_files_path}')
        with open(train_files_path, 'r') as f:
            self.TRAIN_FILES = [line.strip() for line in f]

        test_files_path = os.path.join('data', 'modelnet40_ply_hdf5_2048', 'test_files.txt')
        if not os.path.exists(test_files_path):
            sys.exit(f'not found {test_files_path}')
        with open(test_files_path, 'r') as f:
            self.TEST_FILES = [line.strip() for line in f]


c = Config()

if c.USE_WANDB:
    import wandb

    wandb.config.update(c.__dict__)

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

model = get_pointnet2_model(
    batch_size=c.BATCH_SIZE,
    bn=c.APPLY_BN, bn_momentum=bn_momentum, name='pointnet2_with_ssg', mode='ssg') \
    if c.USE_V2 else \
    get_pointnet1_model(bn=c.APPLY_BN, bn_momentum=bn_momentum, name='pointnet1')

model.build(input_shape=(c.BATCH_SIZE, c.NUM_POINT, 3))
print(model.summary())

# model = get_pointnet1_model(bn=c.APPLY_BN, bn_momentum=bn_momentum, name='pointnet_with_basic')
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
classify_loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_one_epoch():
    train_file_idxs = np.arange(0, len(c.TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fidx in range(len(c.TRAIN_FILES)):
        print("<train in %s>" % c.TRAIN_FILES[train_file_idxs[fidx]])
        train_ds = load_hdf5(c.TRAIN_FILES[train_file_idxs[fidx]], c.NUM_POINT, c.BATCH_SIZE, training=True)

        loss_sum = 0
        total_correct = 0
        total_seen = 0
        num_batches = 0

        for feature, label in train_ds:
            train_loss, train_logits = train_step(point_cloud=feature, labels=label)

            pred = tf.math.argmax(train_logits, axis=1)
            label = tf.squeeze(label)
            correct = np.sum(pred == label, dtype=np.float32)

            with writer.as_default():
                tf.summary.scalar('train/loss', train_loss, step=optimizer.iterations)
                tf.summary.scalar('train/acc', correct / float(c.BATCH_SIZE), step=optimizer.iterations)
                writer.flush()

            total_correct += correct
            total_seen += c.BATCH_SIZE
            loss_sum += train_loss

            num_batches += 1
        mean_loss = loss_sum / num_batches
        accuracy = total_correct / total_seen

        with writer.as_default():
            tf.summary.scalar("train/mean_loss", mean_loss, step=optimizer.iterations)
            tf.summary.scalar("train/mean_accuracy", accuracy, step=optimizer.iterations)
            writer.flush()

        print('train mean loss', mean_loss.numpy())
        print('train accuracy', accuracy)


def eval_one_epoch():
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(c.NUM_CLASSES)]
    total_correct_class = [0 for _ in range(c.NUM_CLASSES)]

    for test_file in c.TEST_FILES:
        test_ds = load_hdf5(test_file, c.NUM_POINT, c.BATCH_SIZE, training=False)

        for feature, label in test_ds:
            logits = val_step(point_cloud=feature)

            pred_val = tf.math.argmax(logits, axis=1)
            label = tf.squeeze(label)
            correct = np.sum(pred_val == label)

            total_correct += correct
            total_seen += c.BATCH_SIZE

            for i in range(label.shape[0]):
                l = label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += tf.cast(pred_val[i] == l, dtype=tf.int64)

    mean_accuracy = total_correct / float(total_seen)
    avg_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))

    with writer.as_default():
        tf.summary.scalar("val/mean_accuracy", mean_accuracy, step=optimizer.iterations)
        tf.summary.scalar("val/avg_class_acc", avg_class_acc, step=optimizer.iterations)
        writer.flush()

    print('val mean accuracy', mean_accuracy)
    print('val avg class acc', avg_class_acc)


@tf.function
def train_step(point_cloud, labels):
    with tf.GradientTape() as tape:
        logits = model(point_cloud, training=True)
        loss = tf.reduce_mean(classify_loss_fn(y_pred=logits, y_true=labels))  # + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss, logits


@tf.function
def val_step(point_cloud):
    logits = model(point_cloud, training=False)
    return logits


def train():
    for epoch in range(c.MAX_EPOCH):
        print("===== Epoch: %03d =====" % epoch)
        with writer.as_default():
            tf.summary.scalar('train/lr', lr, step=epoch)
            tf.summary.scalar('train/bn_momentum', bn_momentum, step=epoch)
            writer.flush()

        train_one_epoch()
        lr.assign(get_decayed_learning_rate(step=optimizer.iterations))
        bn_momentum.assign(get_decayed_bn_momentum(step=optimizer.iterations))
        eval_one_epoch()

        model.save_weights("models/model-" + str(epoch) + "-" + str(optimizer.iterations.numpy()), save_format='tf')


c.load_dataset()
train()
