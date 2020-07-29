#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:05:04 2020

@author: zhuoyin94
"""


import gc
import warnings
import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, Conv2D
from tensorflow.keras.layers import Input, concatenate, Add, ReLU, Flatten
from tensorflow.keras.layers import  GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling1D, AveragePooling1D
from tensorflow.keras import regularizers, constraints, optimizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from scipy.signal import resample

from utils import LoadSave, acc_combo, clf_pred_to_submission, plot_metric
from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk

# np.random.seed(2022)
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


def interp_seq(seq=None, length_interp=61):
    """Interpolating a seq to the fixed length_interp."""
    if len(seq) == length_interp:
        return seq
    interp_df = np.empty((length_interp, seq.shape[1]))
    interp_df[:] = np.nan

    # n_interps = length_interp - len(seq)
    # interp_pos = np.random.randint(1, len(seq)-1, n_interps)
    # interp_df = pd.DataFrame(interp_df, columns=list(seq.columns), index=interp_pos)
    # seq = seq.append(interp_df).sort_index()
    # seq = seq.interpolate(method="polynomial", order=3).reset_index(drop=True)

    for i in range(seq.shape[1]):
        interp_df[:, i] = resample(seq.values[:, i], length_interp)
    return interp_df


def split_seq(seq=None, strides=5, segment_length=40, padding=None):
    """Split the time serie seq according to the strides and segment_length."""
    if len(seq) < (segment_length + strides):
        raise ValueError("The length of seq is less than the segment_length + strides !")
    if padding is not None and padding not in ["zero", "backward"]:
        raise ValueError("Invalid padding method !")

    # Split the time series seq
    seq_split = []
    split_pos = [i for i in list(range(0, len(seq), strides)) if i + segment_length <= len(seq)]

    for pos in split_pos:
        seq_split.append(seq[pos:(pos+segment_length), :])

    # Processing the remain unsplit segments
    if padding is None:
        pass
    elif padding == "backward":
        seq_split.append(seq[(len(seq)-segment_length):, :])
    else:
        seq_tmp = seq[(split_pos[-1]+segment_length):, :]
        n_need_to_pad = segment_length - len(seq_tmp)
        seq_split.append(np.vstack((seq_tmp, np.zeros((n_need_to_pad, seq.shape[1])))))
    return seq_split


def preprocessing_seq(seq=None, length_interp=62, **kwargs):
    """Interpolating a seq on selected feattures to the fixed length_interp"""
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq_val = interp_seq(seq[selected_feats], length_interp=length_interp)
    seq_val = split_seq(seq_val, **kwargs)

    labels = [np.nan] * len(seq_val)
    if "behavior_id" in seq.columns:
        labels = [seq["behavior_id"].iloc[0]] * len(seq_val)
    id_names = [seq["fragment_id"].iloc[0]] * len(seq_val)

    return seq_val, labels, id_names


def build_model(verbose=False, is_compile=True, **kwargs):
    series_length = kwargs.pop("series_length", 61)
    series_feat_size = kwargs.pop("series_feat_size", 8)
    layer_input_series = Input(shape=(series_length, series_feat_size), name="input_series")

    # CONV_2d cross channel
    # -----------------
    layer_reshape = tf.expand_dims(layer_input_series, -1)

    kernel_size_list = [(3, 3), (5, 3), (7, 3), (9, 3), (11, 3), (5, 5)]
    layer_conv_2d_first = []
    for kernel_size in kernel_size_list:
        layer_feat_map = Conv2D(filters=64,
                                kernel_size=kernel_size,
                                activation='relu',
                                padding='same')(layer_reshape)
        layer_residual = ReLU()(layer_feat_map)
        layer_residual = Conv2D(filters=64,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same')(layer_residual)
        layer_0 = Add()([layer_feat_map, layer_residual])
        layer_0 = ReLU()(layer_0)
        layer_conv_2d_first.append(layer_0)

    layer_local_pooling_2d = []
    for layer in layer_conv_2d_first:
        layer_avg_pool = AveragePooling2D(pool_size=(2, 2), padding="valid")(layer)
        layer_avg_pool = Dropout(0.22)(layer_avg_pool)
        layer_local_pooling_2d.append(layer_avg_pool)

    layer_conv_2d_second = []
    for layer in layer_local_pooling_2d:
        layer = Conv2D(filters=128,
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='valid')(layer)
        layer = Dropout(0.22)(layer)
        layer_conv_2d_second.append(layer)

    # Concatenating the pooling layer
    layer_global_pooling_2d = []
    for layer in layer_conv_2d_second:
        layer_global_pooling_2d.append(GlobalAveragePooling2D()(layer))

    # Concat all
    # -----------------
    layer_pooling = concatenate(layer_global_pooling_2d)

    # Output structure
    # -----------------
    layer_output = Dropout(0.22)(layer_pooling)
    layer_output = Dense(128, activation="relu")(layer_output)
    layer_output = Dense(19, activation='softmax')(layer_output)

    model = Model([layer_input_series], layer_output)
    if verbose:
        model.summary()
    if is_compile:
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(0.003, decay=1e-6), metrics=['acc'])
    return model


if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")

    total_data = train_data + test_data
    fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    seq = total_data[14]

    SENDING_TRAINING_INFO = False
    send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)
    INFO_TEXT = "[BEGIN]#Training: {}, #Testing: {}, at: {}".format(
        len(train_data),
        len(test_data),
        str(datetime.now())[:-7])
    send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    ##########################################################################
    # Step 1: Interpolate all the sequence to the fixed length
    # ------------------------
    res = preprocessing_seq(total_data[0].copy())
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(preprocessing_seq, total_data),
                        total=len(total_data)))

    seq_split_data, seq_id, seq_label = [], [], []
    for item in tmp:
        seq_split_data.extend(item[0])
        seq_label.extend(item[1])
        seq_id.extend(item[2])

    total_split_feats = pd.DataFrame(None)
    total_split_feats["behavior_id"] = seq_label
    total_split_feats["fragment_id"] = seq_id
    n_split_train, n_split_test = total_split_feats["behavior_id"].notnull().sum(), total_split_feats["behavior_id"].isnull().sum()

    INFO_TEXT = "[INFO] #splite train: {}, #split test: {}, segment_shape: {}".format(
        n_split_train, n_split_test, len(tmp[0][0]))
    send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    train_seq, test_seq = seq_split_data[:n_split_train], seq_split_data[n_split_train:]
    train_seq, test_seq = np.array(train_seq), np.array(test_seq)

    train_group_id = total_split_feats[total_split_feats["behavior_id"].notnull()]["fragment_id"].values
    test_group_id = total_split_feats[total_split_feats["behavior_id"].isnull()]["fragment_id"].values

    split_labels = total_split_feats[total_split_feats["behavior_id"].notnull()]["behavior_id"].values

    # Preparing and training models
    #########################################################################
    N_FOLDS = 5
    BATCH_SIZE = 2048
    N_EPOCHS = 700
    IS_STRATIFIED = False
    SEED = 2090
    PLOT_TRAINING = False

    folds = GroupKFold(n_splits=N_FOLDS)
    scores = np.zeros((N_FOLDS, 7))
    oof_pred = np.zeros((len(train_seq), 19))
    y_pred = np.zeros((len(test_seq), 19))
    early_stop = EarlyStopping(monitor='val_acc',
                                mode='max',
                                verbose=1,
                                patience=70,
                                restore_best_weights=True)

    # Training the NN classifier
    send_msg_to_dingtalk("\n[INFO]Start training NeuralNets CLASSIFIER at: {}".format(
        str(datetime.now())[:-7]), is_send_msg=SENDING_TRAINING_INFO)
    print("==================================")
    targets_oht = to_categorical(split_labels)
    for fold, (split_tra_id, split_val_id) in enumerate(folds.split(train_seq, targets_oht, train_group_id)):
        d_train, d_valid = train_seq[split_tra_id], train_seq[split_val_id]
        t_train, t_valid = targets_oht[split_tra_id], targets_oht[split_val_id]

        # Destroy all graph nodes in memory
        K.clear_session()
        gc.collect()

        # Training NN classifier
        model = build_model(verbose=False,
                            is_complie=True,
                            series_length=train_seq.shape[1],
                            series_feat_size=train_seq.shape[2])

        history = model.fit(x=[d_train],
                            y=t_train,
                            validation_data=([d_valid], t_valid),
                            callbacks=[early_stop],
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            verbose=2)

        # Model trianing plots
        if PLOT_TRAINING:
            plot_metric(history, metric_type="acc")
            plt.savefig(".//plots//training_fold_acc_{}.png".format(fold),
                        bbox_inches="tight", dpi=500)

            plot_metric(history, metric_type="loss")
            plt.savefig(".//plots//training_fold_loss_{}.png".format(fold),
                        bbox_inches="tight", dpi=500)
            plt.close("all")

        # Training evaluation
        train_pred_proba = model.predict(x=[d_train],
                                          batch_size=BATCH_SIZE)
        valid_pred_proba = model.predict(x=[d_valid],
                                          batch_size=BATCH_SIZE)
        y_pred_proba = model.predict(x=[test_seq],
                                      batch_size=BATCH_SIZE)
        y_pred += y_pred_proba / N_FOLDS
        oof_pred[split_val_id] = valid_pred_proba

        # ---------------
        train_pred_proba = pd.DataFrame(train_pred_proba,
                                        index=train_group_id[split_tra_id])
        valid_pred_proba = pd.DataFrame(valid_pred_proba,
                                        index=train_group_id[split_val_id])
        y_pred_proba = pd.DataFrame(y_pred_proba,
                                    index=test_group_id)
        train_pred_proba = train_pred_proba.groupby(train_pred_proba.index).agg(np.mean).values
        valid_pred_proba = valid_pred_proba.groupby(valid_pred_proba.index).agg(np.mean).values
        y_pred_proba = y_pred_proba.groupby(y_pred_proba.index).agg(np.mean).values

        df_train = pd.DataFrame(None, index=train_group_id[split_tra_id])
        df_valid = pd.DataFrame(None, index=train_group_id[split_val_id])
        df_train["labels"], df_valid["labels"] = split_labels[split_tra_id], split_labels[split_val_id]

        # ---------------
        train_pred_label = np.argmax(train_pred_proba, axis=1).reshape((-1, 1))
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        t_train_label = df_train.groupby(df_train.index).agg(pd.Series.mode).values.reshape((-1, 1))
        t_valid_label = df_valid.groupby(df_valid.index).agg(pd.Series.mode).values.reshape((-1, 1))

        train_f1 = f1_score(
            t_train_label, train_pred_label, average="macro")
        train_acc = accuracy_score(t_train_label,
                                    train_pred_label)
        valid_f1 = f1_score(
            t_valid_label, valid_pred_label, average="macro")
        valid_acc = accuracy_score(t_valid_label,
                                    valid_pred_label)

        train_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_train_label, train_pred_label))).mean()
        valid_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_valid_label, valid_pred_label))).mean()

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_f1, train_acc
        scores[fold, 3], scores[fold, 4] = valid_f1, valid_acc
        scores[fold, 5], scores[fold, 6] = train_custom, valid_custom

        INFO_TEXT = "[INFO] folds {}({}), valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
            fold+1, N_FOLDS, valid_f1, valid_acc, valid_custom)
        send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    oof_pred_label = pd.DataFrame(oof_pred, index=train_group_id)
    oof_pred_label = oof_pred_label.groupby(oof_pred_label.index).agg(np.mean).values
    oof_pred_label = np.argmax(oof_pred_label, axis=1).reshape((-1, 1))

    total_f1 = f1_score(np.array(labels).reshape(-1, 1),
                        oof_pred_label.reshape((-1, 1)), average="macro")
    total_acc = accuracy_score(np.array(labels).reshape(-1, 1),
                               oof_pred_label.reshape((-1, 1)))
    total_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((np.array(labels).reshape((-1, 1)),
                                     oof_pred_label.reshape((-1, 1))))).mean()

    INFO_TEXT = "[INFO] total valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
        total_f1, total_acc, total_custom)
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)
    INFO_TEXT = "[INFO]End training NeuralNets CLASSIFIER at: {}\n".format(
        str(datetime.now())[:-7])
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)
    send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)

    # Saving prediction results
    # ------------------------
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "train_custom", "valid_custom"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(19)])
    y_pred["fragment_id"] = total_split_feats[total_split_feats["behavior_id"].isnull()]["fragment_id"].values
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(19)])
    oof_pred["fragment_id"] = total_split_feats[total_split_feats["behavior_id"].notnull()]["fragment_id"].values
    oof_pred["behavior_id"] = total_split_feats[total_split_feats["behavior_id"].notnull()]["behavior_id"].values

    y_pred_tmp = y_pred.groupby("fragment_id").mean().reset_index()
    oof_pred_tmp = oof_pred.groupby("fragment_id").mean().reset_index()
    clf_pred_to_submission(y_valid=oof_pred_tmp, y_pred=y_pred_tmp, score=scores,
                            target_name="behavior_id", id_name="fragment_id",
                            sub_str_field="nn_split_{}".format(N_FOLDS), save_oof=True)
