#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:19:33 2020

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
from scipy.signal import resample

from utils import LoadSave, acc_combo, clf_pred_to_submission
from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk
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
        ax.plot(seq_interp["time_point"].values, seq_interp[name].values, color="r", marker="s", markersize=5,
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


def preprocessing_seq(seq=None, length_interp=65):
    """Interpolating a seq on selected feattures to the fixed length_interp"""
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq = interp_seq(seq[selected_feats], length_interp=length_interp)
    return seq


if __name__ == "__main__":
    # train_data = load_data("train.pkl")
    # test_data = load_data("test.pkl")


    plot_interp_seq(train_data[66], train_data[1293])
