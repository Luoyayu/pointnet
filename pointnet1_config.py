import os
import sys
import tensorflow as tf


class Config:
    def __init__(self):
        self.MAX_EPOCH: int = 251
        self.BATCH_SIZE = 32
        self.NUM_CLASSES = 40
        self.NUM_POINT = 1024
        self.MAX_NUM_POINT = 2048

        self.BASE_LEARNING_RATE = 0.001
        self.LEARNING_RATE_CLIP = 1e-5
        self.MOMENTUM = 0.99

        self.OPTIMIZER: str = 'adam'
        self.ACTIVATION: str = 'relu'

        self.DECAY_STEP = 7000
        self.DECAY_RATE = 0.7

        self.APPLY_BN = True
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_CLIP = 0.99
        self.BN_DECAY_DECAY_STEP = 7000

        self.TRAIN_FILES: list = []
        self.TEST_FILES: list = []

        self.USE_WANDB = False

        self.KERNEL: str = 'gau'  # gau, min
        self.NKERNEL: int = 300
        self.APPLY_STN = True
        self.STN_Regularization = True
        self.with_std_conv = False
        self.KEEP_PROB = 0.7

        self.INPUT_KEEP_RATE = 0.875

        self.DATA_DIR = 'data'
        self.DATA_NAME = 'modelnet40_ply_hdf5_2048'
        self.writer = tf.summary.create_file_writer("tflogs")

    def load_dataset(self):
        train_files_path = os.path.join(self.DATA_DIR, self.DATA_NAME, 'train_files.txt')
        if not os.path.exists(train_files_path):
            sys.exit(f'not found {train_files_path}')
        with open(train_files_path, 'r') as f:
            self.TRAIN_FILES = [line.strip() for line in f]

        test_files_path = os.path.join(self.DATA_DIR, self.DATA_NAME, 'test_files.txt')
        if not os.path.exists(test_files_path):
            sys.exit(f'not found {test_files_path}')
        with open(test_files_path, 'r') as f:
            self.TEST_FILES = [line.strip() for line in f]
