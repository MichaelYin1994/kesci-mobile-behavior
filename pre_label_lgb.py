#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:31:39 2020

@author: zhuoyin94
"""

import gc
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
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
    stat_feat_fcns = [np.std, np.ptp, np.mean, np.max, np.min]
    for col_name in ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]:
        for fcn in stat_feat_fcns:
            feat_names.append("stat_{}_{}".format(col_name, fcn.__name__))
            feat_vals.append(fcn(seq[col_name]))

    # Step 2: Quantile features
    # feat_name = "acc_x"
    # quantile = np.linspace(0.05, 0.95, 14)
    # feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    # feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
    #                                         feat_name=feat_name))

    # feat_name = "acc_y"
    # quantile = np.linspace(0.05, 0.95, 14)
    # feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    # feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
    #                                         feat_name=feat_name))

    # feat_name = "acc_z"
    # quantile = np.linspace(0.05, 0.95, 3)
    # feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    # feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
    #                                         feat_name=feat_name))

    feat_name = "acc_xg"
    quantile = np.linspace(0.05, 0.95, 7)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_yg"
    quantile = np.linspace(0.05, 0.95, 7)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "acc_zg"
    quantile = np.linspace(0.05, 0.95, 7)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "mod"
    quantile = np.linspace(0.05, 0.95, 8)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    feat_name = "modg"
    quantile = np.linspace(0.05, 0.95, 8)
    feat_names.extend(["seq_{}_quantile_{}".format(feat_name, i) for i in quantile])
    feat_vals.extend(seq_quantile_features(seq, quantile=quantile,
                                            feat_name=feat_name))

    # Step 3: Special count features
    # pos_upper_bound_list = [0.01, 0.5, 1.5, 3, 5, 7] #+ np.linspace(0.05, 1, 3).tolist()
    # pos_lower_bound_list = [-0.01, -0.5, -1.5, -3, -5, -7] #+ (-np.linspace(0.05, 1, 3)).tolist()

    pos_upper_bound_list = np.linspace(0.05, 10, 25).tolist() #+ np.linspace(0.05, 1, 3).tolist()
    pos_lower_bound_list = (-np.linspace(0.05, 10, 25)).tolist() #+ (-np.linspace(0.05, 1, 3)).tolist()

    for low, high in zip(pos_lower_bound_list, pos_upper_bound_list):
        # feat_names.append("between_acc_x_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_x"].between(low, high).sum() / len(seq))
    
        # feat_names.append("between_acc_y_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_y"].between(low, high).sum() / len(seq))
    
        # feat_names.append("between_acc_z_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_z"].between(low, high).sum() / len(seq))

        feat_names.append("between_acc_mod_{}_{}".format(low, high))
        feat_vals.append(seq["mod"].between(low, high).sum() / len(seq))

    # acc_upper_bound_list = [0.01, 1.5, 3, 5, 7] #+ np.linspace(0.05, 7.5, 3).tolist()
    # acc_lower_bound_list = [-0.01, -1.5, -3, -5, -7] #+ (-np.linspace(0.05, 6, 3)).tolist()

    acc_upper_bound_list = np.linspace(0.05, 13, 20).tolist() #+ np.linspace(0.05, 7.5, 3).tolist()
    acc_lower_bound_list = (-np.linspace(0.05, 13, 20)).tolist() #+ (-np.linspace(0.05, 6, 3)).tolist()

    for low, high in zip(acc_lower_bound_list, acc_upper_bound_list):
        # feat_names.append("between_acc_xg_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_xg"].between(low, high).sum() / len(seq))
    
        # feat_names.append("between_acc_yg_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_yg"].between(low, high).sum() / len(seq))
    
        # feat_names.append("between_acc_zg_{}_{}".format(low, high))
        # feat_vals.append(seq["acc_zg"].between(low, high).sum() / len(seq))

        feat_names.append("between_acc_modg_{}_{}".format(low, high))
        feat_vals.append(seq["modg"].between(low, high).sum() / len(seq))

    # Concat all features
    df = pd.DataFrame(np.array(feat_vals).reshape((1, -1)),
                      columns=feat_names)
    return df


if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")

    total_data = train_data + test_data
    fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    seq = total_data[14]

    mapping = {0: 0, 1: 0, 2: 0, 3: 0,
               4: 3, 5: 0, 6: 1, 7: 1,
               8: 1, 9: 1, 10: 1, 11: 0,
               12: 2, 13: 2, 14: 2, 15: 2,
               16: 2, 17: 2, 18: 2}
    labels = [mapping[i] for i in labels]

    total_feats = pd.DataFrame(None)
    total_feats["fragment_id"] = fragment_id
    total_feats["behavior_id"] = labels + [np.nan] * len(test_data)

    ##########################################################################
    # Step 1: Basic stat feature engineering
    # ------------------------
    tmp = stat_feat_seq(seq)
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(stat_feat_seq, total_data),
                        total=len(total_data)))
    stat_feats = pd.concat(tmp, axis=0, ignore_index=True)
    stat_feats["fragment_id"] = fragment_id

    # Step 2: Loading embedding
    # ------------------------
    file_processor = LoadSave()
    embedding_feats = file_processor.load_data(path=".//data_tmp//embedding_df.pkl")

    #########################################################################
    total_feats = pd.merge(total_feats, stat_feats, on="fragment_id", how="left")
    total_feats = pd.merge(total_feats, embedding_feats, on="fragment_id", how="left")

    train_feats = total_feats[total_feats["behavior_id"].notnull()]
    test_feats = total_feats[total_feats["behavior_id"].isnull()].drop("behavior_id", axis=1).reset_index(drop=True)

    n_folds = 5
    scores, importances, oof_pred, y_pred = lightgbm_classifier_training(train_df=train_feats, 
                                                                          test_df=test_feats,
                                                                          id_name="fragment_id",
                                                                          target_name="behavior_id",
                                                                          stratified=True, 
                                                                          shuffle=True,
                                                                          n_classes=4,
                                                                          n_folds=n_folds)
    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                            target_name="behavior_id", id_name="fragment_id",
                            sub_str_field="lgb_pre_label_{}".format(n_folds), save_oof=True)
