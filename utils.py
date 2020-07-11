#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:29:34 2019

@author: yinzhuo
"""

import os
import time
import pickle
import warnings
from memory_profiler import profile
from datetime import datetime
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from scipy import sparse
from numpy import iinfo, finfo, int8, int16, int32, int64, float32, float64
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def timefn(fcn):
    """Decorator for efficency analysis. """
    @wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time


@timefn
def basic_feature_report(data_table=None, precent=None):
    """Reporting basic characteristics of the tabular data data_table."""
    precent = precent or [0.01, 0.25, 0.5, 0.75, 0.95, 0.9995]
    if data_table is None:
        return None
    num_samples = len(data_table)

    # Basic statistics
    basic_report = data_table.isnull().sum()
    basic_report = pd.DataFrame(basic_report, columns=["#missing"])
    basic_report["missing_precent"] = basic_report["#missing"]/num_samples
    basic_report["#uniques"] = data_table.nunique(dropna=False).values
    basic_report["types"] = data_table.dtypes.values
    basic_report.reset_index(inplace=True)
    basic_report.rename(columns={"index":"feature_name"}, inplace=True)

    # Basic quantile of data
    data_description = data_table.describe(precent).transpose()
    data_description.reset_index(inplace=True)
    data_description.rename(columns={"index":"feature_name"}, inplace=True)
    basic_report = pd.merge(basic_report, data_description,
        on='feature_name', how='left')
    return basic_report


class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data


class ReduceMemoryUsage():
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Reduce the memory usage of pandas dataframe.
    
    @Parameters:
    ----------
    data: pandas DataFrame-like
        The dataframe that need to be reduced memory usage.
    verbose: bool
        Whether to print the memory reduction information or not.
        
    @Return:
    ----------
    Memory-reduced dataframe.
    """
    def __init__(self, data_table=None, verbose=True):
        self._data_table = data_table
        self._verbose = verbose

    def type_report(self, data_table):
        """Reporting basic characteristics of the tabular data data_table."""
        data_types = list(map(str, data_table.dtypes.values))
        basic_report = pd.DataFrame(data_types, columns=["types"])
        basic_report["feature_name"] = list(data_table.columns)
        return basic_report

    @timefn
    def reduce_memory_usage(self):
        memory_reduced_data = self.__reduce_memory()
        return memory_reduced_data

    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memory_before_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        types = self.type_report(self._data_table)
        if self._verbose is True:
            print("@Memory usage of data is {:.5f} MB.".format(
                memory_before_reduced))

        # Scan each feature in data_table, reduce the memory usage for features
        for ind, name in enumerate(types["feature_name"].values):
            # ToBeFixed: Unstable query.
            feature_type = str(
                types[types["feature_name"] == name]["types"].iloc[0])

            if (feature_type in "object") and (feature_type in "datetime64[ns]"):
                try:
                    feature_min = self._data_table[name].min()
                    feature_max = self._data_table[name].max()

                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if "int" in feature_type:
                        if feature_min > iinfo(int8).min and feature_max < iinfo(int8).max:
                            self._data_table[name] = self._data_table[name].astype(int8)
                        elif feature_min > iinfo(int16).min and feature_max < iinfo(int16).max:
                            self._data_table[name] = self._data_table[name].astype(int16)
                        elif feature_min > iinfo(int32).min and feature_max < iinfo(int32).max:
                            self._data_table[name] = self._data_table[name].astype(int32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(int64)
                    else:
                        if feature_min > finfo(float32).min and feature_max < finfo(float32).max:
                            self._data_table[name] = self._data_table[name].astype(float32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(float64)
                except Exception as error_msg:
                    print("\n--------ERROR INFORMATION---------")
                    print(error_msg)
                    print("Error on the {}".format(name))
                    print("--------ERROR INFORMATION---------\n")
            if self._verbose is True:
                print("Processed {} feature({}), total is {}.".format(
                    ind + 1, name, len(types)))

        memory_after_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self._verbose is True:
            print("@Memory usage after optimization: {:.5f} MB.".format(
                memory_after_reduced))
            print("@Decreased by {:.5f}%.".format(
                100 * (memory_before_reduced - memory_after_reduced) / memory_before_reduced))
        print("-------------------------------------------")
        return self._data_table


def clf_pred_to_submission(y_valid=None, y_pred=None, score=None, save_oof=True,
                           id_name="fragment_id", target_name="label",
                           sub_str_field=""):
    """Save the oof prediction results to the local path."""
    sub_ind = len(os.listdir(".//submissions//")) + 1
    file_name = "{}_{}_vf1_{}_vacc_{}_vc_{}".format(
        sub_ind, sub_str_field,
        str(round(score["valid_f1"].mean(), 4)).split(".")[1],
        str(round(score["valid_acc"].mean(), 4)).split(".")[1],
        str(round(score["valid_custom"].mean(), 4)).split(".")[1]
        )

    # Saving the submissions.(NOTE: Predicted Class Label +1)
    submission = pd.DataFrame(None)
    submission[id_name] = y_pred[id_name]
    submission["behavior_id"] = np.argmax(
        y_pred.drop([id_name], axis=1).values, axis=1)
    submission.to_csv(".//submissions//{}.csv".format(file_name),
                      header=True, index=False)
    print("@Saving {} to the local.".format(file_name))
    # TODO: Print the ratio of classes
    # print("\n---------------------")
    # pos_precent = len(submission.query("y_pred == 1"))/len(submission) * 100
    # neg_precent = len(submission.query("y_pred == 0"))/len(submission) * 100
    # print("@Submission match precent(1): {:.5f}%, not match precent(0): {:.5f}%".format(
    #     pos_precent, neg_precent))
    # print("---------------------")

    # Saving the oof scores.
    if save_oof:
        y_valid.to_csv(".//submission_oof//{}_valid.csv".format(file_name),
                       index=False, encoding="utf-8")
        y_pred.to_csv(".//submission_oof//{}_pred.csv".format(file_name),
                      index=False, encoding="utf-8")

    # Saving the oof scores.
    if save_oof:
        y_valid.to_csv(".//submission_oof//{}_valid.csv".format(file_name),
                       index=False, encoding="utf-8")
        y_pred.to_csv(".//submission_oof//{}_pred.csv".format(file_name),
                      index=False, encoding="utf-8")


def acc_combo(y_array):
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    y, y_pred = y_array[0], y_array[1]

    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0


def lightgbm_classifier_training(train_df=None, test_df=None, n_folds=5,
                                 n_classes=19, **kwargs):
    """Training a lightgbm binary classifier."""
    lgb_params = {"boosting_type": "gbdt",       # "boosting": "gbdt"
                  "objective": "multiclass",
                  "num_class": n_classes,
                  "metric": "multi_logloss",
                  "num_leaves": 64,
                  "max_depth": 5,
                  "learning_rate": 0.04,
                  "subsample_freq": 1,          # "bagging_freq": 1
                  "subsample": 0.9,             # "bagging_fraction": 0.9
                  "colsample_bytree": 1,        # "feature_fraction": 1
                  "reg_alpha": 0,
                  "reg_lambda": 0.02,
                  "n_jobs": -1,
                  "n_estimators": 5000,
                  "random_state": 255,
                  "verbose": -1}
    id_name = kwargs.pop("id_name", "fragment_id")
    target_name = kwargs.pop("target_name", "behavior_id")
    early_stop_rounds = kwargs.pop("early_stop_rounds", 100)
    random_state = kwargs.pop("random_state", 2022)
    shuffle = kwargs.pop("shuffle", True)
    stratified = kwargs.pop("stratified", True)

    scores, importances, oof_pred, y_pred = lgb_clf_training(
        train=train_df, test=test_df,
        n_folds=n_folds,
        params=lgb_params,
        shuffle=shuffle,
        stratified=stratified,
        n_classes=n_classes,
        early_stop_rounds=early_stop_rounds,
        id_name=id_name,
        target_name=target_name,
        random_state=random_state)
    return scores, importances, oof_pred, y_pred


def lgb_clf_training(train, test, n_folds=5, n_classes=2, shuffle=False,
                     params=None, stratified=False, random_state=2022,
                     early_stop_rounds=100, id_name="user_id", target_name="age"):
    """LightGBM Classifier Training."""
    if stratified == True:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=n_folds, shuffle=shuffle,
                      random_state=random_state)
    X_train = train.drop([id_name, target_name], axis=1)
    X_test = test.drop([id_name], axis=1)
    y_train = train[target_name]

    # Initializing oof prediction and feat importances dataframe
    feat_importance = pd.DataFrame(None)
    feat_importance["feat_name"] = list(X_train.columns)
    scores = np.zeros((n_folds, 8))

    oof_pred = np.zeros((len(train), n_classes))
    y_pred = np.zeros((len(test), n_classes))

    # Training the Lightgbm Classifier
    # ---------------------------------
    print("\n@Start training LightGBM CLASSIFIER at: {}".format(datetime.now()))
    print("==================================")
    print("-- train shape: {}, test shape: {}".format(train.shape, 
                                                      test.shape))
    for fold, (tra_id, val_id) in enumerate(folds.split(X_train, y_train)):
        d_train, d_valid = X_train.iloc[tra_id], X_train.iloc[val_id]
        t_train, t_valid = y_train.iloc[tra_id], y_train.iloc[val_id]
        d_train = lgb.Dataset(d_train, label=t_train)
        d_valid = lgb.Dataset(d_valid, label=t_valid, reference=d_train)

        # Training the model
        clf = lgb.train(params, d_train, valid_sets=d_valid,
                        early_stopping_rounds=early_stop_rounds,
                        verbose_eval=False)

        feat_importance["fold_{}".format(fold+1)] = clf.feature_importance(
            importance_type='split')
        try:
            train_pred_proba = clf.predict(
                X_train.iloc[tra_id], num_iteration=clf.best_iteration)
            valid_pred_proba = clf.predict(
                X_train.iloc[val_id], num_iteration=clf.best_iteration)
            y_pred += clf.predict(X_test, num_iteration=clf.best_iteration)/n_folds
        except:
            train_pred_proba = clf.predict(X_train.iloc[tra_id],
                                           num_iteration=clf.best_iteration)
            train_pred_proba = np.hstack([(1-train_pred_proba).reshape((-1, 1)),
                                          train_pred_proba.reshape((-1, 1))])
            valid_pred_proba = clf.predict(X_train.iloc[val_id],
                                           num_iteration=clf.best_iteration)
            valid_pred_proba = np.hstack([(1-valid_pred_proba).reshape((-1, 1)),
                                          valid_pred_proba.reshape((-1, 1))])
            y_pred_tmp = clf.predict(
                X_test, num_iteration=clf.best_iteration)
            y_pred_tmp = np.hstack([(1-y_pred_tmp).reshape((-1, 1)),
                                    y_pred_tmp.reshape((-1, 1))])
            y_pred += y_pred_tmp/n_folds
        oof_pred[val_id] = valid_pred_proba

        train_pred_label = np.argmax(train_pred_proba, axis=1).reshape((-1, 1))
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        train_f1 = f1_score(
            t_train.values.reshape((-1, 1)), train_pred_label, average="macro")
        train_acc = accuracy_score(t_train.values.reshape((-1, 1)),
                                   train_pred_label)
        valid_f1 = f1_score(
            t_valid.values.reshape((-1, 1)), valid_pred_label, average="macro")
        valid_acc = accuracy_score(t_valid.values.reshape((-1, 1)),
                                   valid_pred_label)

        train_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_train.values.reshape((-1, 1)), train_pred_label))).mean()
        valid_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_valid.values.reshape((-1, 1)), valid_pred_label))).mean()

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_f1, train_acc
        scores[fold, 3], scores[fold, 4] = valid_f1, valid_acc
        scores[fold, 5], scores[fold, 6] = train_custom, valid_custom
        scores[fold, 7] = clf.best_iteration

        print("-- folds {}({}), valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
            fold+1, n_folds, valid_f1, valid_acc, valid_custom))
        params["random_state"] = params["random_state"] + 10086

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
    total_f1 = f1_score(y_train.values.reshape((-1, 1)),
                        oof_pred_label.reshape((-1, 1)), average="macro")
    total_acc = accuracy_score(y_train.values.reshape((-1, 1)),
                               oof_pred_label.reshape((-1, 1)))
    total_custom = np.apply_along_axis(
        acc_combo, 1, np.hstack((y_train.values.reshape((-1, 1)),  oof_pred_label.reshape((-1, 1))))).mean()
    print("-- total valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
        total_f1, total_acc, total_custom))
    print("==================================")
    print("@End training LightGBM CLASSIFIER at: {}\n".format(datetime.now()))

    # TODO: Add the FULL-TRAINING mode.
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "train_custom", "valid_custom",
                                           "best_iters"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(n_classes)])
    y_pred[id_name] = test[id_name]
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(n_classes)])
    oof_pred[id_name], oof_pred[target_name] = train[id_name], train[target_name]
    return scores, feat_importance, oof_pred, y_pred
