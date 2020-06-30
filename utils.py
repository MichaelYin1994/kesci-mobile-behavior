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


def merging_submissions(age_sub_ind=2, gender_sub_ind=1):
    """Merging 2 submission file into 1."""
    sub_ind = len(os.listdir(".//merged_submissions//")) + 1
    file_names = os.listdir(".//submissions//")
    age_file_names = [name for name in file_names if "age" in name]
    gender_file_names = [name for name in file_names if "gender" in name]

    for name in age_file_names:
        splited_name = name.split("_")
        file_ind, score = int(splited_name[0]), int(splited_name[-1][:-4])
        if file_ind == age_sub_ind:
            age_sub = pd.read_csv(".//submissions//{}".format(name))
            age_str = "age({})".format(file_ind) + "_acc_" + str(score)
            age_score = score
            print("-- age sub file name: {}".format(name))
            break

    for name in gender_file_names:
        splited_name = name.split("_")
        file_ind, score = int(splited_name[0]), int(splited_name[-1][:-4])
        if file_ind == gender_sub_ind:
            gender_sub = pd.read_csv(".//submissions//{}".format(name))
            gender_str = "gender({})".format(file_ind) + "_acc_" + str(score)
            gender_score = score
            print("-- gender sub file name: {}".format(name))
            break

    total_score = age_score + gender_score
    merged_sub_name = "{}_{}_{}_acctotal_{}.csv".format(sub_ind, age_str,
                                                        gender_str, total_score)
    print("-- merged file name: {}".format(merged_sub_name))
    total_sub = pd.merge(age_sub, gender_sub, how="left", on="user_id")
    total_sub.to_csv(".//merged_submissions//{}".format(
        merged_sub_name), index=False, encoding="utf-8")
    return total_sub


def clf_pred_to_submission(y_valid=None, y_pred=None, score=None, save_oof=False,
                           id_name="user_id", target_name="age",
                           sub_str_field=""):
    """Save the oof prediction results to the local path."""
    sub_ind = len(os.listdir(".//submissions//")) + 1
    file_name = "{}_{}_tf1_{}_tacc_{}_vf1_{}_vacc_{}".format(
        sub_ind, sub_str_field+"_{}".format(target_name),
        str(round(score["train_f1"].mean(), 4)).split(".")[1],
        str(round(score["train_acc"].mean(), 4)).split(".")[1],
        str(round(score["valid_f1"].mean(), 4)).split(".")[1],
        str(round(score["valid_acc"].mean(), 4)).split(".")[1])

    # Saving the submissions.(NOTE: Predicted Class Label +1)
    submission = pd.DataFrame(None)
    submission[id_name] = y_pred[id_name]
    submission["predicted_{}".format(target_name)] = np.argmax(
        y_pred.drop([id_name], axis=1).values, axis=1) + 1
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


def lightgbm_age_classifier_training(train_df=None, test_df=None, n_folds=5,
                                     **kwargs):
    """Training a classifier for the age."""
    lgb_params = {"boosting_type": "gbdt",   # boosting="gbdt"
                  "objective": "multiclass",
                  "num_class": 10,
                  "num_leaves": 64,
                  "max_depth": 4,
                  "learning_rate": 0.13,
                  "subsample_freq": 1,        # bagging_freq=1
                  "subsample": 0.95,          # bagging_fraction=0.9
                  "colsample_bytree": 0.95,   # feature_fraction=0.9
                  "reg_alpha": 0,
                  "reg_lambda": 1.72,
                  "n_jobs": -1,
                  "n_estimators": 5000,
                  "random_state": 956,
                  "verbose": -1}
    scores, importances, oof_pred, y_pred = lgb_clf_training(
        train=train_df, test=test_df, n_folds=n_folds,
        params=lgb_params, n_classes=10,
        id_name="user_id", target_name="age", **kwargs)
    return scores, importances, oof_pred, y_pred


def lightgbm_age_regressor_training(train_df=None, test_df=None, n_folds=5,
                                    **kwargs):
    """Training a lightGBM regressor."""
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l1"},
        "n_estimators": 5000,
        "num_leaves": 31,
        "max_depth": 4,
        "learning_rate": 0.06,
        "colsample_bytree": 0.9,               # feature_fraction=0.9
        "subsample": 0.9,                      # bagging_fraction=0.8
        "subsample_freq": 1,                   # bagging_freq=1
        "reg_alpha": 0,
        "reg_lambda": 0.01,
        "random_state": 2022,
        "n_jobs": -1,
        "verbose": -1}
    scores, importances, oof_pred, y_pred = lgb_reg_training(
        train=train_df, test=test_df,
        params=lgb_params, n_folds=n_folds, **kwargs)
    oof_pred_ret, y_pred_ret = oof_pred.copy(), y_pred.copy()

    oof_pred["oof_pred"] = np.clip(oof_pred["oof_pred"].values, 0, 9)
    oof_pred["oof_pred"] = np.round(oof_pred["oof_pred"])
    oof_pred["oof_pred"] = oof_pred["oof_pred"].astype(int)

    y_pred["y_pred"] = np.clip(y_pred["y_pred"].values, 0, 9)
    y_pred["y_pred"] = np.round(y_pred["y_pred"].values)
    y_pred["y_pred"] = y_pred["y_pred"].astype(int)

    valid_f1 = f1_score(oof_pred["oof_pred"].values.reshape((-1, 1)),
                        oof_pred["age"].values.reshape((-1, 1)),
                        average="macro")
    valid_acc = accuracy_score(oof_pred["oof_pred"].values.reshape((-1, 1)),
                               oof_pred["age"].values.reshape((-1, 1)))
    print("-- LOCAL: Regression oof f1: {:.5f}, accuracy: {:.5f}".format(
        valid_f1, valid_acc))
    return scores, importances, oof_pred_ret, y_pred_ret


def lightgbm_age_regressor_training_sparse(train_sp=None, test_sp=None,
                                           train_targets=None, **kwargs):
    """LightGBM training for the regression problem using the sparse matrix."""
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l1",
        "n_estimators": 5000,
        "num_leaves": 31,
        "max_depth": 4,
        "learning_rate": 0.06,
        "colsample_bytree": 0.9,               # feature_fraction=0.9
        "subsample": 0.9,                      # bagging_fraction=0.8
        "subsample_freq": 1,                   # bagging_freq=1
        "reg_alpha": 0,
        "reg_lambda": 0.01,
        "random_state": 2022,
        "n_jobs": -1,
        "verbose": -1}

    scores, importances, oof_pred, y_pred = lgb_reg_training_sparse(
        train_sp=train_sp,
        test_sp=test_sp,
        train_targets=train_targets,
        params=lgb_params, **kwargs)
    oof_pred_ret, y_pred_ret = oof_pred.copy(), y_pred.copy()

    oof_pred["oof_pred"] = np.clip(oof_pred["oof_pred"].values, 0, 9)
    oof_pred["oof_pred"] = np.round(oof_pred["oof_pred"])
    oof_pred["oof_pred"] = oof_pred["oof_pred"].astype(int)

    y_pred["y_pred"] = np.clip(y_pred["y_pred"].values, 0, 9)
    y_pred["y_pred"] = np.round(y_pred["y_pred"].values)
    y_pred["y_pred"] = y_pred["y_pred"].astype(int)

    valid_f1 = f1_score(oof_pred["oof_pred"].values.reshape((-1, 1)),
                        oof_pred["age"].values.reshape((-1, 1)),
                        average="macro")
    valid_acc = accuracy_score(oof_pred["oof_pred"].values.reshape((-1, 1)),
                               oof_pred["age"].values.reshape((-1, 1)))
    print("-- LOCAL: Regression oof f1: {:.5f}, accuracy: {:.5f}".format(
        valid_f1, valid_acc))
    return scores, importances, oof_pred_ret, y_pred_ret


def lightgbm_age_classifier_training_sparse(train_sp=None, test_sp=None,
                                            train_targets=None, n_folds=5,
                                            **kwargs):
    """Training a classifier for the age."""
    lgb_params = {"boosting_type": "gbdt",   # boosting="gbdt"
                  "objective": "multiclass",
                  "num_class": 10,
                  "num_leaves": 64,
                  "max_depth": 4,
                  "learning_rate": 0.1,
                  "subsample_freq": 1,        # bagging_freq=1
                  "subsample": 0.95,          # bagging_fraction=0.9
                  "colsample_bytree": 0.95,   # feature_fraction=0.9
                  "reg_alpha": 0,
                  "reg_lambda": 1.72,
                  "n_jobs": -1,
                  "n_estimators": 5000,
                  "random_state": 956,
                  "verbose": -1}
    scores, importances, oof_pred, y_pred = lgb_clf_training_sparse(
        train=train_sp,
        test=test_sp,
        train_targets=train_targets,
        params=lgb_params,
        n_folds=n_folds,
        target_name="age",
        n_classes=10,
        **kwargs)
    return scores, importances, oof_pred, y_pred


def lightgbm_gender_classifier_training_sparse(train_sp=None, test_sp=None,
                                               train_targets=None, n_folds=5,
                                               **kwargs):
    """Training a classifier for the age."""
    lgb_params = {"boosting_type": "gbdt",   # boosting="gbdt"
                  "objective": "cross_entropy",
                  "metric": "binary_error",
                  "num_leaves": 32,
                  "max_depth": 4,
                  "learning_rate": 0.1,
                  "subsample_freq": 1,        # bagging_freq=1
                  "subsample": 0.95,          # bagging_fraction=0.9
                  "colsample_bytree": 0.95,   # feature_fraction=0.9
                  "reg_alpha": 0,
                  "reg_lambda": 0.2,
                  "n_jobs": -1,
                  "n_estimators": 5000,
                  "random_state": 777,
                  "verbose": -1}
    scores, importances, oof_pred, y_pred = lgb_clf_training_sparse(
        train=train_sp,
        test=test_sp,
        train_targets=train_targets,
        params=lgb_params,
        n_folds=n_folds,
        target_name="gender",
        n_classes=2,
        **kwargs)
    return scores, importances, oof_pred, y_pred


def lgb_reg_training_sparse(train_sp=None, test_sp=None, train_targets=None,
                            **kwargs):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    LightGBM Regressor for CSR data matirx training. 

    @Parameters:
    ----------
    train_sp: {csr-matrix}
        Training data in the form of CSR matrix.
    test_sp: {csr-matirx}
        Testing data in the form of CSR matrix.
    train_targets: {array-like}
        Training target, it is a numpy array

    @Return:
    ----------
    Regression results.
    """
    # Initializing parameters
    id_name = kwargs.pop("id_name", "user_id")
    target_name = kwargs.pop("target_name", "age")
    train_id_vals = kwargs.pop("train_id_vals", None)
    test_id_vals = kwargs.pop("test_id_vals", None)

    n_folds = kwargs.pop("n_folds", 5)
    params = kwargs.pop("params", None)
    shuffle = kwargs.pop("shuffle", True)
    feat_names = kwargs.pop("feat_names", None)
    random_state = kwargs.pop("random_state", 2022)
    early_stop_rounds = kwargs.pop("early_stop_rounds", 100)

    if train_id_vals is None or test_id_vals is None or train_targets is None:
        raise ValueError("Invalid id columns or target columns !")
    if sparse.issparse(train_sp) != True or sparse.issparse(test_sp) != True:
        raise ValueError("Inputs are dense matrix, while sparse matrixs are required !")

    if params is None:
        raise ValueError("Invalid training parameters !")
    if feat_names is None:
        feat_names = np.arange(train_sp.shape[1])

    # Initializing oof prediction and feat importances dataframe
    folds = KFold(
        n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    feat_importance = pd.DataFrame(None)
    feat_importance["feat_name"] = feat_names
    scores = np.zeros((n_folds, 6))

    oof_pred = np.zeros((train_sp.shape[0], ))
    y_pred = np.zeros((test_sp.shape[0], ))

    # Training the Lightgbm Regressor
    # ---------------------------------
    print("\n@Start training LightGBM REGRESSOR(Sparse) at: {}".format(datetime.now()))
    print("==================================")
    print("-- train_sp shape: {}, test_sp shape: {}".format(train_sp.shape, test_sp.shape))
    for fold, (tra_id, val_id) in enumerate(folds.split(train_sp, train_targets)):
        d_train, d_valid = train_sp[tra_id], train_sp[val_id]
        t_train, t_valid = train_targets[tra_id], train_targets[val_id]

        # Training the model
        reg = lgb.LGBMRegressor(**params)
        reg.fit(d_train, t_train, eval_set=[(d_valid, t_valid)],
                early_stopping_rounds=early_stop_rounds, verbose=False)

        feat_importance["fold_{}".format(fold+1)] = reg.feature_importances_
        train_pred = reg.predict(d_train, num_iteration=reg.best_iteration_)
        valid_pred = reg.predict(d_valid, num_iteration=reg.best_iteration_)
        oof_pred[val_id] = valid_pred
        y_pred += reg.predict(test_sp, num_iteration=reg.best_iteration_)/n_folds

        train_mse = mean_squared_error(
            t_train.reshape((-1, 1)), train_pred.reshape((-1, 1)))
        train_mae = mean_absolute_error(
            t_train.reshape((-1, 1)), train_pred.reshape((-1, 1)))
        valid_mse = mean_squared_error(
            t_valid.reshape((-1, 1)), valid_pred.reshape((-1, 1)))
        valid_mae = mean_absolute_error(
            t_valid.reshape((-1, 1)), valid_pred.reshape((-1, 1)))

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_mse, train_mae
        scores[fold, 3], scores[fold, 4] = valid_mse, valid_mae
        scores[fold, 5] = reg.best_iteration_

        print("-- folds {}({}), train MSE: {:.5f}, MAE: {:.5f}; valid MSE: {:.5f}, MAE: {:.5f}".format(
            fold+1, n_folds, train_mse, train_mae, valid_mse, valid_mae))
        params["random_state"] += 1000

    total_mse = mean_squared_error(train_targets.reshape((-1, 1)),
                                   oof_pred.reshape((-1, 1)))
    total_mae = mean_absolute_error(train_targets.reshape((-1, 1)),
                                    oof_pred.reshape((-1, 1)))
    print("-- total valid MSE: {:.5f}, MAE: {:.5f}".format(
        total_mse, total_mae))
    print("==================================")
    print("@End training LightGBM REGRESSOR at: {}\n".format(datetime.now()))
    # TODO: Add the FULL-TRAINING mode.
    scores = pd.DataFrame(scores, columns=["folds", "train_mse", "train_mae",
                                           "valid_mse", "valid_mae", "best_iters"])
    y_pred = pd.DataFrame(y_pred, columns=["y_pred"])
    y_pred[id_name] = test_id_vals
    oof_pred = pd.DataFrame(oof_pred, columns=["oof_pred"])
    oof_pred[id_name], oof_pred[target_name] = train_id_vals, train_targets
    return scores, feat_importance, oof_pred, y_pred


def lgb_clf_training_sparse(train=None, test=None, train_targets=None, **kwargs):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    LightGBM Classifier for CSR data matirx training. 
    
    @Parameters:
    ----------
    train: {csr-matrix}
        Training data in the form of CSR matrix.
    test: {csr-matirx}
        Testing data in the form of CSR matrix.
    train_targets: {array-like}
        Training target, it is a numpy array

    @Return:
    ----------
    Classification results.
    """
    id_name = kwargs.pop("id_name", "user_id")
    target_name = kwargs.pop("target_name", "age")
    train_id_vals = kwargs.pop("train_id_vals", None)
    test_id_vals = kwargs.pop("test_id_vals", None)

    if train_id_vals is None or test_id_vals is None or train_targets is None:
        raise ValueError("Invalid id columns or target columns !")
    if sparse.issparse(train) != True or sparse.issparse(test) != True:
        raise ValueError("Inputs are dense matrix, while sparse matrixs are required !")

    # Initializing parameters
    n_folds = kwargs.pop("n_folds", 5)
    params = kwargs.pop("params", None)
    shuffle = kwargs.pop("shuffle", True)
    n_classes = kwargs.pop("n_classes", 2)
    stratified = kwargs.pop("stratified", False)
    feat_names = kwargs.pop("feat_names", None)
    random_state = kwargs.pop("random_state", 2022)
    early_stop_rounds = kwargs.pop("early_stop_rounds", 100)

    if params is None:
        raise ValueError("Invalid training parameters !")
    if feat_names is None:
        feat_names = np.arange(train.shape[1])

    # Preparing
    if stratified == True:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=n_folds, shuffle=shuffle,
                      random_state=random_state)

    # Initializing oof prediction and feat importances dataframe
    feat_importance = pd.DataFrame(None)
    feat_importance["feat_name"] = feat_names
    scores = np.zeros((n_folds, 6))

    oof_pred = np.zeros((train.shape[0], n_classes))
    y_pred = np.zeros((test.shape[0], n_classes))

    # Training the Lightgbm Classifier
    # ---------------------------------
    print("\n@Start training LightGBM CLASSIFIER(CSR) at:{}".format(datetime.now()))
    print("==================================")
    print("-- train shape: {}, test shape: {}".format(train.shape, test.shape))
    for fold, (tra_id, val_id) in enumerate(folds.split(train, train_targets)):
        d_train, d_valid = train[tra_id], train[val_id]
        t_train, t_valid = train_targets[tra_id], train_targets[val_id]

        # Training the model
        clf = lgb.LGBMClassifier(**params)
        clf.fit(d_train, t_train, eval_set=[(d_valid, t_valid)],
                early_stopping_rounds=early_stop_rounds, verbose=False)

        feat_importance["fold_{}".format(fold+1)] = clf.feature_importances_
        train_pred_proba = clf.predict_proba(
            d_train, num_iteration=clf.best_iteration_)
        valid_pred_proba = clf.predict_proba(
            d_valid, num_iteration=clf.best_iteration_)
        y_pred += clf.predict_proba(
            test, num_iteration=clf.best_iteration_)/n_folds

        oof_pred[val_id] = valid_pred_proba
        train_pred_label = np.argmax(train_pred_proba, axis=1).reshape((-1, 1))
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        train_f1 = f1_score(
            t_train.reshape((-1, 1)), train_pred_label, average="macro")
        train_acc = accuracy_score(t_train.reshape((-1, 1)),
                                   train_pred_label)
        valid_f1 = f1_score(
            t_valid.reshape((-1, 1)), valid_pred_label, average="macro")
        valid_acc = accuracy_score(t_valid.reshape((-1, 1)),
                                   valid_pred_label)

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_f1, train_acc
        scores[fold, 3], scores[fold, 4] = valid_f1, valid_acc
        scores[fold, 5] = clf.best_iteration_

        print("-- folds {}({}), valid f1: {:.5f}, acc: {:.5f}".format(
            fold+1, n_folds, valid_f1, valid_acc))
        params["random_state"] = params["random_state"] + 10086

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
    total_f1 = f1_score(train_targets.reshape((-1, 1)),
                        oof_pred_label.reshape((-1, 1)), average="macro")
    total_acc = accuracy_score(train_targets.reshape((-1, 1)),
                               oof_pred_label.reshape((-1, 1)))
    print("-- total valid f1: {:.5f}, acc: {:.5f}".format(
        total_f1, total_acc))
    print("==================================")
    print("@End training LightGBM CLASSIFIER at:{}\n".format(datetime.now()))
    # TODO: Add the FULL-TRAINING mode.
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "best_iters"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(n_classes)])
    y_pred[id_name] = test_id_vals
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(n_classes)])
    oof_pred[id_name], oof_pred[target_name] = train_id_vals, train_targets
    return scores, feat_importance, oof_pred, y_pred


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
    scores = np.zeros((n_folds, 6))

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

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_f1, train_acc
        scores[fold, 3], scores[fold, 4] = valid_f1, valid_acc
        scores[fold, 5] = clf.best_iteration

        print("-- folds {}({}), valid f1: {:.5f}, acc: {:.5f}".format(
            fold+1, n_folds, valid_f1, valid_acc))
        params["random_state"] = params["random_state"] + 10086

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
    total_f1 = f1_score(y_train.values.reshape((-1, 1)),
                        oof_pred_label.reshape((-1, 1)), average="macro")
    total_acc = accuracy_score(y_train.values.reshape((-1, 1)),
                               oof_pred_label.reshape((-1, 1)))
    print("-- total valid f1: {:.5f}, acc: {:.5f}".format(
        total_f1, total_acc))
    print("==================================")
    print("@End training LightGBM CLASSIFIER at: {}\n".format(datetime.now()))
    # TODO: Add the FULL-TRAINING mode.
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "best_iters"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(n_classes)])
    y_pred[id_name] = test[id_name]
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(n_classes)])
    oof_pred[id_name], oof_pred[target_name] = train[id_name], train[target_name]
    return scores, feat_importance, oof_pred, y_pred


def lgb_reg_training(train, test, params=None, n_folds=5, random_state=2022,
                     id_name="user_id", target_name="age",
                     early_stop_rounds=100, shuffle=True):
    """Lightgbm Regressor for regression task."""
    # Shuffle the DataFrame
    X_train = train.drop([id_name, target_name], axis=1)
    X_test = test.drop([id_name], axis=1)
    y_train = train[target_name]
    folds = KFold(
        n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    # Initializing oof prediction and feat importances dataframe
    feat_importance = pd.DataFrame(None)
    feat_importance["feat_name"] = list(X_train.columns)
    scores = np.zeros((n_folds, 6))

    oof_pred = np.zeros((len(train), ))
    y_pred = np.zeros((len(test), ))

    # Training the Lightgbm Regressor
    # ---------------------------------
    print("\n@Start training LightGBM REGRESSOR at: {}".format(datetime.now()))
    print("==================================")
    print("-- train shape: {}, test shape: {}".format(train.shape, 
                                                      test.shape))
    for fold, (tra_id, val_id) in enumerate(folds.split(X_train)):
        d_train, d_valid = X_train.iloc[tra_id], X_train.iloc[val_id]
        t_train, t_valid = y_train.iloc[tra_id], y_train.iloc[val_id]
        d_train = lgb.Dataset(d_train, label=t_train)
        d_valid = lgb.Dataset(d_valid, label=t_valid, reference=d_train)

        # Training the model
        reg = lgb.train(params, d_train, valid_sets=d_valid,
                        early_stopping_rounds=early_stop_rounds,
                        verbose_eval=False)

        feat_importance["fold_{}".format(fold)] = reg.feature_importance(
            importance_type='split')
        train_pred = reg.predict(X_train.iloc[tra_id],
                                 num_iteration=reg.best_iteration)
        valid_pred = reg.predict(X_train.iloc[val_id],
                                 num_iteration=reg.best_iteration)
        oof_pred[val_id] = valid_pred
        y_pred += reg.predict(X_test, num_iteration=reg.best_iteration)/n_folds

        train_mse = mean_squared_error(
            t_train.values.reshape((-1, 1)), train_pred.reshape((-1, 1)))
        train_mae = mean_absolute_error(
            t_train.values.reshape((-1, 1)), train_pred.reshape((-1, 1)))
        valid_mse = mean_squared_error(
            t_valid.values.reshape((-1, 1)), valid_pred.reshape((-1, 1)))
        valid_mae = mean_absolute_error(
            t_valid.values.reshape((-1, 1)), valid_pred.reshape((-1, 1)))

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_mse, train_mae
        scores[fold, 3], scores[fold, 4] = valid_mse, valid_mae
        scores[fold, 5] = reg.best_iteration

        print("-- folds {}({}), train MSE: {:.5f}, MAE: {:.5f}; valid MSE: {:.5f}, MAE: {:.5f}".format(
            fold+1, n_folds, train_mse, train_mae, valid_mse, valid_mae))
        params["random_state"] += 1000

    total_mse = mean_squared_error(train[target_name].values.reshape((-1, 1)),
                                   oof_pred.reshape((-1, 1)))
    total_mae = mean_absolute_error(train[target_name].values.reshape((-1, 1)),
                                   oof_pred.reshape((-1, 1)))
    print("-- total valid MSE: {:.5f}, MAE: {:.5f}".format(
        total_mse, total_mae))
    print("==================================")
    print("@End training LightGBM REGRESSOR at: {}\n".format(datetime.now()))
    # TODO: Add the FULL-TRAINING mode.
    scores = pd.DataFrame(scores, columns=["folds", "train_mse", "train_mae",
                                           "valid_mse", "valid_mae", "best_iters"])
    y_pred = pd.DataFrame(y_pred, columns=["y_pred"])
    y_pred[id_name] = test[id_name]
    oof_pred = pd.DataFrame(oof_pred, columns=["oof_pred"])
    oof_pred[id_name], oof_pred[target_name] = train[id_name], train[target_name]
    return scores, feat_importance, oof_pred, y_pred
