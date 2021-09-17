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
from numpy.core.shape_base import block
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import signal
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers.normalization import LayerNormalization
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
        oht_label = np.zeros((19, ), dtype=np.int32)
        oht_label[label] = 1

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

    # Transform to the tf_fcn object
    tf_fcn = lambda f_path: tf.py_function(
        fcn, [f_path], (tf.float32, tf.int32)
    )

    def set_shape(full_name):
        seq, oht_label = tf_fcn(full_name)

        seq.set_shape((target_length, 8))
        oht_label.set_shape((19, ))

        return seq, oht_label

    return set_shape


def resnet_block_conv1d(seq, n_filters, kernel_size):
    '''ResNet-like CONV-1D block.'''
    # Increase dimension
    x = tf.keras.layers.Conv1D(
        filters=n_filters, kernel_size=1, padding='same', activation='relu'
    )(seq)
    x = tf.keras.layers.LayerNormalization()(x)

    # Feature extraction
    x = tf.keras.layers.Conv1D(
        filters=n_filters, kernel_size=kernel_size, padding='same', activation='relu'
    )(seq)
    x = tf.keras.layers.LayerNormalization()(x)

    # Increase dimension
    x = tf.keras.layers.Conv1D(
        filters=int(n_filters * 4), kernel_size=kernel_size,
        padding='same', activation='relu'
    )(seq)
    x = tf.keras.layers.LayerNormalization()(x)

    # Residual connection
    x = tf.keras.layers.Add()([seq, x])

    return x


def block_cascade(x, kernel_size=5):
    '''Cascade the ResNet-like Conv-1D blocks'''
    # STEP 1: Block cascade
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=1, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = resnet_block_conv1d(x, 32, kernel_size)

    x = tf.keras.layers.Conv1D(
        filters=256, kernel_size=1, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = resnet_block_conv1d(x, 64, kernel_size)

    # STEP 2: Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    return x


def build_model(verbose=False, is_compile=True, **kwargs):
    '''
    Build and complie a Conv-1D based ResNet-like neural network.

    @References:
    --------------
    [1] https://github.com/blueloveTH/xwbank2020_baseline_keras/blob/master/models.py
    [2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
    [3] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
    [4] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).

    @Return:
    --------------
    Keras model object
    '''

    # Input parameters
    # --------
    ts_length = kwargs.pop('ts_length', 61)
    ts_dim = kwargs.pop('ts_dim', 8)
    learning_rate = kwargs.pop('learning_rate', 0.0001)

    layer_ts_input = tf.keras.layers.Input(
        shape=(ts_length, ts_dim), name='layer_input'
    )

    # Structure
    # --------
    block_output_list = []
    for kernel_size in [3, 5, 7]:
        block_output_list.append(
            block_cascade(
                layer_ts_input, kernel_size=kernel_size
            )
        )

    block_output = tf.keras.layers.concatenate(
        block_output_list
    )

    x = tf.keras.layers.Dense(512, activation='relu')(block_output)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    layer_output = tf.keras.layers.Dense(19, activation='softmax')(x)

    # Build and complie model
    # --------
    model = tf.keras.models.Model(
        [layer_ts_input], layer_output
    )

    if verbose:
        model.summary()
    if is_compile:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['acc']
        )

    return model

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

    IS_RANDOM_VISUALIZING = True
    IS_SEND_MSG_TO_DINGTALK = False

    # Training preparing
    # **********************
    total_file_name_list = os.listdir(TRAIN_PATH_NAME)
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

        train_ts_ds = train_ts_ds.batch(BATCH_SIZE)
        train_ts_ds = train_ts_ds.prefetch(buffer_size=int(BATCH_SIZE * 8))
        valid_ts_ds = valid_ts_ds.batch(BATCH_SIZE)
        valid_ts_ds = valid_ts_ds.prefetch(buffer_size=int(BATCH_SIZE * 8))

        # build & train model
        # --------
        model = build_model(
            ts_length=SEGMENT_LENGTH,
            ts_dim=8,
            learning_rate=MODEL_LR
        )

        history = model.fit(
            train_ts_ds,
            epochs=NUM_EPOCHS,
            validation_data=valid_ts_ds,
            callbacks=callbacks,
            verbose=1
        )

        # Evaulate the preformance
        # --------

    print('==================================')
