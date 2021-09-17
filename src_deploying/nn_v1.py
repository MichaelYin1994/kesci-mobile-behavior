#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109171758
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
Training a Neural Network using tf.data.Dataset API.
'''

import gc
import multiprocessing as mp
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import signal
from sklearn.model_selection import KFold
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk

GLOBAL_RANDOM_SEED = 2912
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

warnings.filterwarnings('ignore')

TASK_NAME = 'kesci_2020'
GPU_ID = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit the visible GPUs
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU')

        # Limit the GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)
###############################################################################
def custom_eval_metric(y_true, y_pred):
    pass


def random_crop():
    '''Random crop a segment of a time series.'''
    pass


def random_shift():
    '''Random shift time series.'''
    pass


def load_preprocess_single_ts(stage, target_length):
    '''Load and process a single time series data.'''

    def fcn(full_name):
        if full_name is None:
            raise ValueError('Invalid data file name !')

        # Load the data
        # --------
        full_name = bytes.decode(full_name.numpy())

        df = pd.read_csv(full_name)
        df['mod'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['modg'] = np.sqrt(df['acc_xg']**2 + df['acc_yg']**2 + df['acc_zg']**2)

        # Get the label and interpolate time series to the fixed length
        # --------
        seq = df.drop(['fragment_id', 'behavior_id', 'time_point'], axis=1).values

        label = df['behavior_id'].values[0]
        oht_label = np.zeros((1, 19), dtype=np.int32)
        oht_label[0, label] = 1

        if len(seq) != target_length:
            interp_seq = np.zeros((target_length, seq.shape[1]))

            for i in range(seq.shape[1]):
                interp_seq[:, i] = signal.resample(
                    seq[:, i], target_length
                )
            seq = interp_seq

        # Augment the time series
        # --------
        if stage == 'train':
            pos = np.random.randint(5, target_length, 1)[0]
            seq[pos:] = 0.0

        seq = tf.convert_to_tensor(
            seq, dtype=tf.float32
        )
        oht_label = tf.convert_to_tensor(
            oht_label, dtype=tf.int32
        )

        return seq, oht_label

    tf_fcn = lambda f_path: tf.py_function(
        fcn, [f_path], (tf.float32, tf.int32)
    )

    return tf_fcn


def build_model():
    pass


if __name__ == '__main__':
    # Global parameters
    # **********************
    BATCH_SIZE = 128
    NUM_EPOCHS = 256
    EARLY_STOP_ROUNDS = 6
    N_FOLDS = 5
    TTA_ROUNDS = 20

    SEGMENT_LENGTH = 61

    TRAIN_PATH_NAME = '../data_tmp/train_separated_csv/'
    MODEL_NAME = 'ResNet50_dataaug_rtx3090'
    MODEL_LR = 0.0003
    MODEL_LABEL_SMOOTHING = 0

    IS_RANDOM_VISUALIZING = False
    IS_SEND_MSG_TO_DINGTALK = False

    # Training preparing
    # **********************
    total_file_name_list = os.listdir(TRAIN_PATH_NAME)[:128]
    total_file_name_list = [
        os.path.join(TRAIN_PATH_NAME, item) for item in total_file_name_list
    ]

    # Random plot a time series
    if IS_RANDOM_VISUALIZING:
        load_process_train_ts = load_preprocess_single_ts(
            stage='train', target_length=SEGMENT_LENGTH
        )

        train_path_ds = tf.data.Dataset.from_tensor_slices(total_file_name_list)
        train_ts_ds = train_path_ds.map(
            load_process_train_ts, num_parallel_calls=mp.cpu_count()
        )

        for item in train_ts_ds.take(1):
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.grid(True)
            ax.set_xlim(0, SEGMENT_LENGTH+1)
            ax.set_xlabel('Timestamp', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(
                'Class label: {}'.format(item[1].numpy()),
                fontsize=10)
            ax.tick_params(axis="both", labelsize=10)

            tmp = item[0].numpy()
            for i in range(8):
                ax.plot(
                    np.arange(0, SEGMENT_LENGTH), tmp[:, i],
                    linestyle='--', linewidth=2, marker='o', markersize=4
                )
            plt.tight_layout()

    # Various callbacks
    # --------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode='max',
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.7,
                patience=20,
                min_lr=0.000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=os.path.join(TASK_NAME, MODEL_NAME),
            gpu_id=GPU_ID),
    ]

    # Fold strategy
    # --------
    fold = KFold(
            n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_RANDOM_SEED
        )

    # Training start
    # **********************
    print('\n[INFO] {} NN training start...'.format(
        str(datetime.now())[:-4]))
    print('==================================')
    for train_idx, valid_idx in fold.split(total_file_name_list, y=None):
        # Destroy all graph nodes in GPU memory
        # --------
        K.clear_session()
        gc.collect()

        # Construct the train & valid path names
        # --------
        train_file_name_list = [
            total_file_name_list[i] for i in train_idx
        ]
        valid_file_name_list = [
            total_file_name_list[i] for i in valid_idx
        ]

        train_path_ds = tf.data.Dataset.from_tensor_slices(
            train_file_name_list
        )
        valid_path_ds = tf.data.Dataset.from_tensor_slices(
            valid_file_name_list
        )

        # Construct the loading pipeline
        # --------
        load_process_train_ts = load_preprocess_single_ts(
            stage='train', target_length=SEGMENT_LENGTH
        )
        load_process_valid_ts = load_preprocess_single_ts(
            stage='valid', target_length=SEGMENT_LENGTH
        )

        train_ts_ds = train_path_ds.map(
            load_process_train_ts, num_parallel_calls=mp.cpu_count()
        )
        valid_ts_ds = train_path_ds.map(
            load_process_valid_ts, num_parallel_calls=mp.cpu_count()
        )

        # build & train model
        # --------
        model = build_model()


        # Evaulate the preformance
        # --------

    print('==================================')
