#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:09:06 2020

@author: zhuoyin94
"""

import pandas as pd
from utils import LoadSave

if __name__ == "__main__":
    train = pd.read_csv(".//data//sensor_train.csv")
    test = pd.read_csv(".//data//sensor_test.csv")
    sub = pd.read_csv(".//data//submission.csv")

    train["fragment_id"] = train["fragment_id"] + 10000000
    train_unique_id = train["fragment_id"].unique()
    test_unique_id = test["fragment_id"].unique()

    # train_data_list = []
    # for i in train_unique_id:
    #     tmp_data = train.query("fragment_id == {}".format(i))
    #     tmp_data.reset_index(drop=True, inplace=True)
    #     train_data_list.append(tmp_data)

    # test_data_list = []
    # for i in test_unique_id:
    #     tmp_data = test.query("fragment_id == {}".format(i))
    #     tmp_data.reset_index(drop=True, inplace=True)
    #     test_data_list.append(tmp_data)

    # file_processor = LoadSave()
    # file_processor.save_data(path=".//data_tmp//train.pkl", data=train_data_list)
    # file_processor.save_data(path=".//data_tmp//test.pkl", data=test_data_list)
