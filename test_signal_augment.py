#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:33:34 2020

@author: zhuoyin94
"""


import numpy as np
from utils import LoadSave
from dingtalk_remote_monitor import send_msg_to_dingtalk
from scipy.signal import resample
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    for i in range(seq.shape[1]):
        interp_df[:, i] = resample(seq.values[:, i], length_interp)
    return interp_df


# def plot_list_seq(seq_list=None):
#     feat_names = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg"]

#     fig, ax_objs = plt.subplots(6, 1, figsize=(12, 12))
#     ax_objs = ax_objs.ravel()

#     for ind, name in enumerate(feat_names):
#         ax = ax_objs[ind]
#         ax.plot(seq["time_point"].values, seq[name].values, color="k", marker="o", markersize=5,
#                 linewidth=2, linestyle="--", label=name)
#         ax.plot(seq_interp["time_point"].values, seq_interp[name].values, color="red", marker="s", markersize=1.8,
#                 linewidth=2, linestyle=" ", label=name)

#         ax.set_xlim(0,  )
#         ax.tick_params(axis="both", labelsize=8)
#         ax.grid(True)
#         ax.legend(fontsize=8, loc='best')
#     plt.tight_layout()


def plot_seq_compare(seq, seq_compare):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(seq, color="k", marker="o", markersize=5,
            linewidth=2, linestyle="--", label="seq")
    ax.plot(seq_compare, color="red", marker="s", markersize=1.8,
            linewidth=2, linestyle="-", label="seq_compare")
    ax.set_xlim(0,  )
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True)
    ax.legend(fontsize=8, loc='best')


def preprocessing_seq(seq=None, length_interp=65):
    """Interpolating a seq on selected feattures to the fixed length_interp"""
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq = interp_seq(seq[selected_feats], length_interp=length_interp)
    return seq


def stretch(x, target_length=65):
    l = int(target_length * (1 + (np.random.random() - 0.5) / 3))

    y = np.zeros(x.shape)
    for i in range(x.shape[1]):
        y_tmp = resample(x[:, i], l)
        if l < target_length:
            y[:len(y_tmp), i] = y_tmp
        else:
            y[:, i] = y_tmp[:target_length]
    return y


def amplify(x, target_length=65):
    alpha = (np.random.random() - 0.5)
    factor = -alpha * x + (1 + alpha)
    return x * factor


def white_noise():
    pass


# def augment(x, target_length=61):
#     result = np.zeros(shape= (4, 61))
#     for i in range(3):
#         if np.random.random() < 0.33:
#             new_y = stretch(x)
#         elif np.random.random() < 0.66:
#             new_y = amplify(x)
#         else:
#             new_y = stretch(x)
#             new_y = amplify(new_y)
#         result[i, :] = new_y
#     return result





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
    # res = preprocessing_seq(total_data[0].copy())
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(preprocessing_seq, total_data),
    #                     total=len(total_data)))
    # train_seq, test_seq = tmp[:len(train_data)], tmp[len(train_data):]
    # train_seq, test_seq = np.array(train_seq), np.array(test_seq)

    plt.close("all")
    ind = 123
    plot_seq_compare(train_seq[ind][:, 1], stretch(train_seq[ind])[:, 1])
    plot_seq_compare(train_seq[ind][:, 1], amplify(train_seq[ind])[:, 1])
    
