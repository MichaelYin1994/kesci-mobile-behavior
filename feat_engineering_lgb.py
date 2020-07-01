#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:46:07 2020

@author: zhuoyin94
"""

import os
import multiprocessing as mp
import pywt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import LoadSave, lightgbm_classifier_training, clf_pred_to_submission

np.random.seed(202)
##############################################################################
def load_train():
    file_processor = LoadSave()
    return file_processor.load_data(path=".//data_tmp//train.pkl")


def load_valid():
    file_processor = LoadSave()
    return file_processor.load_data(".//data_tmp//valid.pkl")


def compute_signal_mat_stat(signal_mat=None):
    """Compute the mean and std of each channel."""
    feat_names, feat_vals = [], []

    feat_vals.extend(np.mean(signal_mat, axis=1).tolist())
    feat_names.extend(["c_mean_{}".format(i) for i in range(128)])

    feat_vals.extend(np.std(signal_mat, axis=1).tolist())
    feat_names.extend(["c_std_{}".format(i) for i in range(128)])

    # Concat to pandas DataFrame
    feat_vals = np.array(feat_vals).reshape(1, -1)
    df = pd.DataFrame(feat_vals, columns=feat_names)
    return df


if __name__ == "__main__":
    # Loading all data and preparing the dataframe
    train = load_train()
    valid = load_valid()

    total_feat = pd.DataFrame(None)
    total_signal_mat = train[1] + valid[1]
    total_feat["id"] = train[0] + valid[0]
    total_feat = pd.merge(total_feat, train[2], how="left", on="id")


    # STEP 1: Basic stat feature engineering
    with mp.Pool(processes=mp.cpu_count()) as p:
        stat_feat = list(tqdm(p.imap(compute_signal_mat_stat, total_signal_mat),
                              total=len(total_signal_mat)))
    stat_feat = pd.concat(stat_feat, axis=0).reset_index(drop=True)


    # ---------------
    total_feat = total_feat.join(stat_feat)

    train_feat = total_feat[total_feat["label"].notnull()]
    valid_feat = total_feat[total_feat["label"].isnull()].drop("label", axis=1).reset_index(drop=True)

    scores, importances, oof_pred, y_pred = lightgbm_classifier_training(train_df=train_feat, 
                                                                          test_df=valid_feat,
                                                                          n_folds=5)
    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                            target_name="label", id_name="id",
                            sub_str_field="lgb_5")
