#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109201930
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
Model online inference.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow import keras
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from utils import LiteModel, custom_eval_metric, njit_f1, tf_custom_eval

GLOBAL_RANDOM_SEED = 2912
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

warnings.filterwarnings('ignore')

TASK_NAME = 'kesci_2020'
SERVED_MODEL = 'nn_v1'
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
if SERVED_MODEL == 'nn_v1':
    from nn_v1 import build_model, preprocess_single_ts


def load_quantize_model(model_path_list, is_quantization):
    '''Load models from model_path_list and quantize models'''
    model_list = []

    for model_path in model_path_list:
        model = tf.keras.models.load_model(model_path)

        if is_quantization:
            model = LiteModel.from_keras_model(model)
        model_list.append(model)

    return model_list


def preprocess_single_ts_df(ts_df, stage, target_length):
    '''Preprocess a single time series in DataFrame format'''
    ts_df['mod'] = np.sqrt(
        ts_df['acc_x']**2 + ts_df['acc_y']**2 + ts_df['acc_z']**2
    )
    ts_df['modg'] = np.sqrt(
        ts_df['acc_xg']**2 + ts_df['acc_yg']**2 + ts_df['acc_zg']**2
    )

    # Preprocess
    # --------
    if 'behavior_id' in ts_df.columns:
        seq = ts_df.drop(
            ['fragment_id', 'behavior_id', 'time_point'], axis=1
        ).values
    else:
        seq = ts_df.drop(
            ['fragment_id', 'time_point'], axis=1
        ).values       

    seq = preprocess_single_ts(
        seq, stage, target_length
    )

    # Convert the ts to the Tensorflow tensor object
    # --------
    # seq = tf.convert_to_tensor(
    #     seq, dtype=tf.float32
    # )

    return seq


class InferServer():
    def __init__(self, model_list=None):
        self.model_list = model_list

    def preprocess(self, **kwargs):
        '''Processing data.'''
        preprocessed_ts = preprocess_single_ts_df(**kwargs)
        return preprocessed_ts

    def predict_proba(self, ts):
        '''Make inference.'''

        # Make prediction
        predicted_proba = 0

        for model in self.model_list:
            predicted_proba += model.predict(ts) / len(self.model_list)

        return predicted_proba

    def predict(self, ts):
        '''Make inference.'''

        # Make prediction
        predicted_proba = 0

        for model in self.model_list:
            predicted_proba += model.predict(ts) / len(self.model_list)

        # Label generation
        predicted_label = np.argmax(
            predicted_proba, axis=1
        )

        return predicted_label


if __name__ == '__main__':
    # Global parameters
    # **********************
    TTA_ROUNDS = 5
    SEGMENT_LENGTH = 61
    IS_QUANTIZATION = True
    TEST_PATH_NAME = '../data_tmp/test_separated_csv/'

    # Loading models & testing data
    # **********************
    model_path = os.path.join(
        '../models/', SERVED_MODEL
    )
    model_path_list = [
        os.path.join(model_path, item) for item in os.listdir(model_path)
    ]
    model_list = load_quantize_model(
        model_path_list, IS_QUANTIZATION
    )

    # Loading the meta data
    # **********************
    test_file_name_list = os.listdir(TEST_PATH_NAME)

    # Meta DataFrame
    test_meta_df = pd.DataFrame(None)
    test_meta_df['fragment_id'] = [
        int(item.split('_')[1]) for item in test_file_name_list
    ]
    test_meta_df['behavior_id'] = [
        int(item[:-4].split('_')[-1]) for item in test_file_name_list
    ]
    test_file_name_list = [
        os.path.join(TEST_PATH_NAME, item) for item in test_file_name_list
    ]

    # Make inference
    # **********************
    server = InferServer(
        model_list=model_list
    )
    predicted_label_list = []

    for file_name in tqdm(test_file_name_list):
        # Stage 1: Fetch data from a source
        # --------
        ts_df = pd.read_csv(file_name)

        # Stage 2: Make inference
        # --------
        predicted_proba_list_tmp = []

        for curr_rounds in range(TTA_ROUNDS + 1):
            ts_processed = server.preprocess(
                ts_df=ts_df, stage='train', target_length=SEGMENT_LENGTH
            )
            ts_processed = np.reshape(
                ts_processed, [-1] + list(ts_processed.shape)
            ).astype(np.float32)
            predicted_proba_list_tmp.append(
                server.predict_proba(ts_processed)
            )

        predicted_proba_list_tmp = np.mean(
            predicted_proba_list_tmp, axis=0
        )
        predicted_label_list_tmp = np.argmax(
            predicted_proba_list_tmp, axis=1
        )

        # Stage 3: Send out the prediction result
        # --------
        predicted_label_list.append(predicted_label_list_tmp)

    # Make final evaluation
    # **********************
    test_meta_df['predicted_label'] = predicted_label_list
    y_pred_label = np.array(predicted_label_list).ravel()

    y_pred_label_oht = tf.keras.utils.to_categorical(
        y_pred_label
    )
    y_total_label_oht = tf.keras.utils.to_categorical(
        test_meta_df['behavior_id'].values
    )

    test_f1 = f1_score(
        test_meta_df['behavior_id'].values,
        y_pred_label.reshape(-1, 1),
        average='macro'
    )
    test_acc = accuracy_score(
        test_meta_df['behavior_id'].values,
        y_pred_label.reshape(-1, 1)
    )
    test_custom = custom_eval_metric(
        y_total_label_oht, y_pred_label_oht
    )

    print('\nTESTING EVAULATION RESULTS:')
    print('*****************')
    print(
        '-- {} test f1: {:.5f}, acc {:5f}, custom: {:.5f}'.format(
            str(datetime.now())[:-4],
            test_f1, test_acc, test_custom,
        )
    )
    print('*****************')
