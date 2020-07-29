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
    oof_pred_names = ["32_nn_split_15_vf1_7812_vacc_792_vc_8214_valid.csv",
                      "31_nn_split_15_vf1_7747_vacc_7851_vc_8163_valid.csv",
                      "24_nn_15_vf1_7685_vacc_781_vc_8124_valid.csv"]
    y_pred_names = ["32_nn_split_15_vf1_7812_vacc_792_vc_8214_pred.csv",
                    "31_nn_split_15_vf1_7747_vacc_7851_vc_8163_pred.csv",
                    "24_nn_15_vf1_7685_vacc_781_vc_8124_pred.csv"]

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

    oof_pred = pd.concat(oof_pred, axis=1)
    oof_pred["fragment_id"], oof_pred["behavior_id"] = oof_id_col, oof_target_col

    y_pred = pd.concat(y_pred, axis=1)
    y_pred["fragment_id"] = y_pred_id_col

    # Model training
    ##########################################################################
    n_folds = 10
    scores, importances, oof_pred, y_pred = lightgbm_classifier_training(train_df=oof_pred, 
                                                                          test_df=y_pred,
                                                                          id_name="fragment_id",
                                                                          target_name="behavior_id",
                                                                          stratified=True, 
                                                                          shuffle=True,
                                                                          n_classes=19,
                                                                          n_folds=n_folds)
    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                            target_name="behavior_id", id_name="fragment_id",
                            sub_str_field="lgb_stack_{}".format(n_folds), save_oof=False)
