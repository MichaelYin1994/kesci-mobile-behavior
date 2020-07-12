#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:00:03 2020

@author: zhuoyin94
"""

import gc
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, LSTM, concatenate, \
    Bidirectional, GRU, Dropout, BatchNormalization, Conv1D, LayerNormalization
from tensorflow.keras import regularizers, constraints, optimizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import LoadSave
from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk

sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(2022)
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
###############################################################################
def load_data(name=None):
    """Load data from .//data_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//data_tmp//" + name)
    return data


def plot_seq(seq=None, seq_interp=None):
    feat_names = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg"]

    fig, ax_objs = plt.subplots(6, 1, figsize=(10, 10))
    ax_objs = ax_objs.ravel()

    for ind, name in enumerate(feat_names):
        ax = ax_objs[ind]
        ax.plot(seq["time_point"].values, seq[name].values, color="k", marker="o", markersize=5,
                linewidth=1.8, linestyle="-", label=name)
        ax.plot(seq_interp["time_point"].values, seq_interp[name].values, color="r", marker="s", markersize=3,
                linewidth=1.1, linestyle="--", label=name)
        # ax.set_xlim(0, len(seq))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True)
        ax.legend(fontsize=8, loc='best')
    plt.tight_layout()


def interp_seq(seq=None, length_interp=61):
    """Interpolating a seq to the fixed length_interp."""
    if len(seq) == length_interp:
        return seq



if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")

    total_data = train_data + test_data
    fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    seq = total_data[14]

    total_feats = pd.DataFrame(None)
    total_feats["fragment_id"] = fragment_id
    total_feats["behavior_id"] = labels + [np.nan] * len(test_data)

    ##########################################################################
    seq = total_data[544]
    interp_feats = ["time_point", "acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg"]
    seq_tmp = seq[interp_feats].copy().reset_index(drop=True)
    length_interp, length_seq = 61, len(seq)

    n_interps = length_interp - length_seq
    interp_df = np.empty((n_interps, 7))
    interp_df[:] = np.nan
    interp_index = np.random.randint(1, length_interp-1, n_interps)

    interp_df = pd.DataFrame(interp_df, columns=interp_feats, index=interp_index)
    seq_tmp = seq_tmp.append(interp_df).sort_index()
    seq_tmp = seq_tmp.interpolate(method="linear")

    plot_seq(seq, seq_tmp)

    # plot_seq(seq)