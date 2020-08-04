#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:37:02 2020

@author: gv1001107
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
if __name__ == "__main__":
    # oof_pred_names = ["56_nn_split_10_vf1_8033_vacc_8125_vc_8391_valid.csv",
    #                   "51_nn_split_10_vf1_8077_vacc_8154_vc_8421_valid.csv"]
    # y_pred_names = ["56_nn_split_10_vf1_8033_vacc_8125_vc_8391_pred.csv",
    #                 "51_nn_split_10_vf1_8077_vacc_8154_vc_8421_pred.csv"]

    oof_pred_names = ["68_nn_5_vf1_8022_vacc_8098_vc_8372_valid.csv",
                      "69_lgb_stack_5_vf1_815_vacc_8221_vc_8479_valid.csv"]
    y_pred_names = ["68_nn_5_vf1_8022_vacc_8098_vc_8372_pred.csv",
                    "69_lgb_stack_5_vf1_815_vacc_8221_vc_8479_pred.csv"]

    oof_pred, y_pred = [], []
    for oof_pred_name, y_pred_name in zip(oof_pred_names, y_pred_names):
        ind = int(oof_pred_name.split("_")[0])

        # oof processing
        oof_pred_df = pd.read_csv(".//submission_oof//"+oof_pred_name)
        oof_pred_col_names = list(oof_pred_df.columns)
        oof_pred_col_names = {name:str(ind)+"_"+name for name in oof_pred_col_names if name not in ["behavior_id", "fragment_id"]}
        oof_pred_df.rename(oof_pred_col_names, axis=1, inplace=True)

        oof_id_col = oof_pred_df["fragment_id"].values
        oof_target_col = oof_pred_df["behavior_id"].values
        oof_pred_df.drop(["fragment_id", "behavior_id"], axis=1, inplace=True)
        oof_pred.append(oof_pred_df)

        # target processing
        y_pred_df = pd.read_csv(".//submission_oof//"+y_pred_name)
        y_pred_col_names = list(y_pred_df.columns)
        y_pred_col_names = {name:str(ind)+"_"+name for name in y_pred_col_names if name not in ["behavior_id", "fragment_id"]}
        y_pred_df.rename(y_pred_col_names, axis=1, inplace=True)

        y_pred_id_col = y_pred_df["fragment_id"].values
        y_pred_df.drop(["fragment_id"], axis=1, inplace=True)
        y_pred.append(y_pred_df)

    # oof_pred = pd.concat(oof_pred, axis=1)
    # oof_pred["fragment_id"], oof_pred["behavior_id"] = oof_id_col, oof_target_col

    # y_pred = pd.concat(y_pred, axis=1)
    # y_pred["fragment_id"] = y_pred_id_col

    # Averaging
    oof_pred = np.mean([item.values for item in oof_pred], axis=0)
    y_pred = np.mean([item.values for item in y_pred], axis=0)

    oof_pred = pd.DataFrame(oof_pred)
    y_pred = pd.DataFrame(y_pred)
    oof_pred["fragment_id"], oof_pred["behavior_id"] = oof_id_col, oof_target_col
    y_pred["fragment_id"] = y_pred_id_col

    scores = np.zeros((5, 8))
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                            "valid_f1", "valid_acc",
                                            "train_custom", "valid_custom",
                                            "best_iters"])

    # Model training
    ##########################################################################
    n_folds = 10
    # scores, importances, oof_pred, y_pred = lightgbm_classifier_training(train_df=oof_pred, 
    #                                                                       test_df=y_pred,
    #                                                                       id_name="fragment_id",
    #                                                                       target_name="behavior_id",
    #                                                                       stratified=True, 
    #                                                                       shuffle=True,
    #                                                                       n_classes=19,
    #                                                                       n_folds=n_folds)
    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                            target_name="behavior_id", id_name="fragment_id",
                            sub_str_field="lgb_stack_{}".format(n_folds), save_oof=True)
