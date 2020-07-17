#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:39:02 2020

@author: zhuoyin94
"""

import gc
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.signal import savgol_filter, resample
from utils import LoadSave, lightgbm_classifier_training, clf_pred_to_submission

np.random.seed(1080)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_data(name=None):
    """Load data from .//data_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//data_tmp//" + name)
    return data


def seq_quantile_features(seq, quantile=None, feat_name="x"):
    """Quantile statistics for a specified feature."""
    if quantile is None:
        quantile = [0.1, 0.15, 0.25]
    if len(seq) == 0:
        return [0] * len(quantile)

    feat_vals = []
    for qu in quantile:
        feat_vals.append(seq[feat_name].quantile(qu))
    return feat_vals


def stat_feat_seq(seq=None):
    """Basic feature engineering for each dim"""
    feat_vals, feat_names = [], []

    # Preparing: mod quantile feats
    # https://github.com/ycd2016/xw2020_cnn_baseline/blob/master/baseline.py
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    # Step 1: Basic stat features of each column
    stat_feat_fcns = [np.ptp, np.std, np.min, np.max, np.mean, np.median]
    for col_name in ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]:
        for fcn in stat_feat_fcns:
            feat_names.append("stat_{}_{}".format(col_name, fcn.__name__))
            feat_vals.append(fcn(seq[col_name]))

    # Step 2: Quantile features
    feat_name = "acc_x"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_y"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_z"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_xg"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_yg"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_zg"
    quantile = np.linspace(0.02, 0.99, 17)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "mod"
    quantile = np.linspace(0.1, 0.95, 7)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "modg"
    quantile = np.linspace(0.1, 0.95, 7)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    # Step 3: Special count features
    pos_upper_bound_list = [0.01] #+ np.linspace(0.05, 1, 3).tolist()
    pos_lower_bound_list = [-0.01] #+ (-np.linspace(0.05, 1, 3)).tolist()
    for low, high in zip(pos_lower_bound_list, pos_upper_bound_list):
        feat_names.append("between_acc_x_{}_{}".format(low, high))
        feat_vals.append(seq["acc_x"].between(low, high).sum() / len(seq))
    
        feat_names.append("between_acc_y_{}_{}".format(low, high))
        feat_vals.append(seq["acc_y"].between(low, high).sum() / len(seq))
    
        feat_names.append("between_acc_z_{}_{}".format(low, high))
        feat_vals.append(seq["acc_z"].between(low, high).sum() / len(seq))


    acc_upper_bound_list = [0.01] #+ np.linspace(0.05, 7.5, 3).tolist()
    acc_lower_bound_list = [-0.01] #+ (-np.linspace(0.05, 6, 3)).tolist()
    for low, high in zip(acc_lower_bound_list, acc_upper_bound_list):
        feat_names.append("between_acc_xg_{}_{}".format(low, high))
        feat_vals.append(seq["acc_xg"].between(low, high).sum() / len(seq))
    
        feat_names.append("between_acc_yg_{}_{}".format(low, high))
        feat_vals.append(seq["acc_yg"].between(low, high).sum() / len(seq))
    
        feat_names.append("between_acc_zg_{}_{}".format(low, high))
        feat_vals.append(seq["acc_zg"].between(low, high).sum() / len(seq))

    # Concat all features
    df = pd.DataFrame(np.array(feat_vals).reshape((1, -1)),
                      columns=feat_names)
    return df


def plot_interp_seq(seq=None, seq_interp=None):
    feat_names = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg"]

    fig, ax_objs = plt.subplots(6, 1, figsize=(12, 12))
    ax_objs = ax_objs.ravel()

    for ind, name in enumerate(feat_names):
        ax = ax_objs[ind]
        ax.plot(seq[name].values, color="k", marker="o", markersize=5,
                linewidth=1.8, linestyle="--", label=name)
        ax.plot(seq_interp["time_point"].values, seq_interp[name].values, color="r", marker="s", markersize=3,
                linewidth=1.1, linestyle=" ", label=name)

        tmp_vals = resample(seq[name].values, len(seq_interp))
        ax.plot(seq_interp["time_point"].values, tmp_vals, color="g", marker="^", markersize=3,
                linewidth=1.8, linestyle="--", label="Resample")

        tmp_vals = savgol_filter(tmp_vals, window_length=7,
                                 polyorder=3)
        ax.plot(tmp_vals, color="b", marker="^", markersize=3,
                linewidth=1.8, linestyle="--", label="S-G filter")

        ax.set_xlim(0,  )
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
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq = interp_seq(seq[selected_feats], length_interp=length_interp)

    return seq.values


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

    # ##########################################################################
    # # Step 1: Basic stat feature engineering
    # # ------------------------
    # tmp = stat_feat_seq(seq)
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(stat_feat_seq, total_data),
    #                     total=len(total_data)))
    # stat_feats = pd.concat(tmp, axis=0, ignore_index=True)
    # stat_feats["fragment_id"] = fragment_id

    # # Step 2: Loading embedding
    # # ------------------------
    # file_processor = LoadSave()
    # embedding_feats = file_processor.load_data(path=".//data_tmp//embedding_df.pkl")

    ###########################################################################
    '''
    Plot 1: Random sequence visualizing.
    '''
    seq = total_data[1104]
    seq_interp = interp_seq(seq.copy(), 200)
    plot_interp_seq(seq, seq_interp)