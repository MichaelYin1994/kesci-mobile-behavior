#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:00:03 2020

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
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import  GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras import regularizers, constraints, optimizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

from utils import LoadSave, acc_combo, clf_pred_to_submission
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


def plot_interp_seq(seq=None, seq_interp=None):
    feat_names = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg"]

    fig, ax_objs = plt.subplots(6, 1, figsize=(14, 10))
    ax_objs = ax_objs.ravel()

    for ind, name in enumerate(feat_names):
        ax = ax_objs[ind]
        ax.plot(seq["time_point"].values, seq[name].values, color="k", marker="o", markersize=5,
                linewidth=1.8, linestyle="-", label=name)
        ax.plot(seq_interp["time_point"].values, seq_interp[name].values, color="r", marker="s", markersize=3,
                linewidth=1.1, linestyle=" ", label=name)
        # ax.set_xlim(0, len(seq))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True)
        ax.legend(fontsize=8, loc='best')
    plt.tight_layout()


def interp_seq(seq=None, length_interp=61):
    """Interpolating a seq to the fixed length_interp."""
    if len(seq) == length_interp:
        return seq

    n_interps = length_interp - len(seq)
    interp_df = np.empty((n_interps, seq.shape[1]))
    interp_df[:] = np.nan

    interp_pos = np.random.randint(1, len(seq)-1, n_interps)
    interp_df = pd.DataFrame(interp_df, columns=list(seq.columns), index=interp_pos)
    seq = seq.append(interp_df).sort_index()
    seq = seq.interpolate(method="linear").reset_index(drop=True)
    return seq


def preprocessing_seq(seq=None, length_interp=61):
    """Interpolating a seq on selected feattures to the fixed length_interp"""
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq = interp_seq(seq[selected_feats], length_interp=length_interp)
    return seq.values


def build_model_baseline(verbose=False, is_compile=True, **kwargs):
    """Baseline deep learning model.

    Referneces:
    --------
    [1] https://github.com/ycd2016/xw2020_cnn_baseline/blob/master/baseline.py#L28
    """
    input_layer = Input(shape=(61, 8))

    X = tf.expand_dims(input_layer, -1)
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = MaxPooling2D()(X)

    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = Conv2D(filters=512,
               kernel_size=(5, 5),
               activation='relu',
               padding='same')(X)
    X = GlobalMaxPooling2D()(X)
    X = Dropout(0.2)(X)
    X = Dense(19, activation='softmax')(X)

    model = Model([input_layer], X)
    if verbose:
        model.summary()
    if is_compile:
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(0.002), metrics=['acc'])
    return model


def build_model(verbose=False, is_compile=True, **kwargs):
    input_layer = Input(shape=(61, 8), name="input_layer")

    conv_layer = Conv1D(64, kernel_size=4, padding="valid",
                        kernel_initializer="he_uniform")(input_layer)
    avg_pool_conv = GlobalAveragePooling1D()(conv_layer)
    max_pool_conv = GlobalMaxPooling1D()(conv_layer)

    gru_layer = Bidirectional(GRU(32, return_sequences=True))(input_layer)
    avg_pool_gru = GlobalAveragePooling1D()(gru_layer)
    max_pool_gru = GlobalMaxPooling1D()(gru_layer)

    # Concate all layer
    # -----------
    layer_total = concatenate([avg_pool_conv,
                               max_pool_conv,
                               avg_pool_gru,
                               max_pool_gru])
    dense_layer = Dense(32, activation="relu")(layer_total)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dropout(0.17)(dense_layer)
    layer_pred = Dense(19, activation='softmax',
                       name="output")(dense_layer)

    model = Model([input_layer], layer_pred)
    if verbose:
        model.summary()
    if is_compile:
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(0.005), metrics=['acc'])
    return model


if __name__ == "__main__":
    # train_data = load_data("train.pkl")
    # test_data = load_data("test.pkl")

    # total_data = train_data + test_data
    # fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    # labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    # seq = total_data[14]

    # total_feats = pd.DataFrame(None)
    # total_feats["fragment_id"] = fragment_id
    # total_feats["behavior_id"] = labels + [np.nan] * len(test_data)
    # total_feats["is_train"] = [True] * len(train_data) + [False] * len(test_data)

    # SENDING_TRAINING_INFO = False
    # send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)
    # INFO_TEXT = "[BEGIN]#Training: {}, #Testing: {}, at: {}".format(
    #     len(total_feats.query("is_train == True")),
    #     len(total_feats.query("is_train == False")),
    #     str(datetime.now())[:-7])
    # send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    # ##########################################################################
    # # Step 1: Interpolate all the sequence to the fixed length
    # # ------------------------
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(preprocessing_seq, total_data),
    #                     total=len(total_data)))
    # train_seq, test_seq = np.array(tmp[:len(train_data)]), np.array(tmp[len(train_data):])

    # Preparing and training models
    #########################################################################
    N_FOLDS = 5
    BATCH_SIZE = 4096
    N_EPOCHS = 200
    IS_STRATIFIED = False
    SEED = 1024

    if IS_STRATIFIED:
        folds = StratifiedKFold(n_splits=N_FOLDS,
                                shuffle=True,
                                random_state=SEED)
    else:
        folds = KFold(n_splits=N_FOLDS,
                      shuffle=True,
                      random_state=SEED)

    scores = np.zeros((N_FOLDS, 7))
    oof_pred = np.zeros((len(train_data), 19))
    y_pred = np.zeros((len(test_data), 19))
    early_stop = EarlyStopping(monitor='val_acc',
                                mode='max',
                                verbose=1,
                                patience=20,
                                restore_best_weights=True)

    # Training the NN classifier
    send_msg_to_dingtalk("\n[INFO]Start training NeuralNets CLASSIFIER at: {}".format(
        str(datetime.now())[:-7]), is_send_msg=SENDING_TRAINING_INFO)
    print("==================================")
    targets_oht = to_categorical(labels)
    for fold, (tra_id, val_id) in enumerate(folds.split(train_seq, targets_oht)):
        d_train, d_valid = train_seq[tra_id], train_seq[val_id]
        t_train, t_valid = targets_oht[tra_id], targets_oht[val_id]

        # Destroy all graph nodes in memory
        K.clear_session()
        gc.collect()

        # Training NN classifier
        model = build_model_baseline(verbose=False, is_complie=True)

        model.fit(x=[d_train],
                  y=t_train,
                  validation_data=([d_valid], t_valid),
                  callbacks=[early_stop],
                  batch_size=BATCH_SIZE,
                  epochs=N_EPOCHS,
                  verbose=2)

        train_pred_proba = model.predict(x=[d_train],
                                          batch_size=BATCH_SIZE)
        valid_pred_proba = model.predict(x=[d_valid],
                                          batch_size=BATCH_SIZE)
        y_pred_proba = model.predict(x=[test_seq],
                                      batch_size=BATCH_SIZE)
        y_pred += y_pred_proba / N_FOLDS

        oof_pred[val_id] = valid_pred_proba
        train_pred_label = np.argmax(train_pred_proba, axis=1).reshape((-1, 1))
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        t_train_label = np.argmax(t_train, axis=1).reshape((-1, 1))
        t_valid_label = np.argmax(t_valid, axis=1).reshape((-1, 1))

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

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
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
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "train_custom", "valid_custom"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(19)])
    y_pred["fragment_id"] = total_feats.query("is_train == False")["fragment_id"].values
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(19)])
    oof_pred["fragment_id"], oof_pred["behavior_id"] = total_feats.query("is_train == True")["fragment_id"].values, total_feats.query("is_train == True")["behavior_id"].values

    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                           target_name="behavior_id", id_name="fragment_id",
                           sub_str_field="lgb_{}".format(N_FOLDS), save_oof=False)
