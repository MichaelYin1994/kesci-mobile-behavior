#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:18:53 2020

@author: zhuoyin94
"""
import os
import multiprocessing as mp
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tslearn.metrics import dtw
from _ucrdtw import ucrdtw
from utils import LoadSave

np.random.seed(202)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
##############################################################################
def read_train_label():
    return pd.read_csv(".//data//Training Set//train_label.csv")


def read_person_data_csv(person_data_name):
    signal_mat = pd.read_csv(person_data_name, header=None).values
    return signal_mat


def processing_train_data():
    PATH = ".//data//Training Set//train_data//"
    train_label = read_train_label()
    train_label["id"] = train_label["file_name"].apply(lambda x: x[:-4])
    train_label.drop("file_name", axis=1, inplace=True)

    person_fold_names = os.listdir(PATH)
    person_fold_names = sorted(person_fold_names, key=lambda x: (x[0], int(x[1:])))

    train_data, train_index = [], []
    for fold_name in person_fold_names:
        person_data_names = os.listdir(PATH + fold_name)
        person_data_names = sorted(person_data_names,
                                   key=lambda c: int(c[:-4].split("-")[-1]))

        train_index.extend([name[:-4] for name in person_data_names])
        person_data_names = [PATH + fold_name + "//" + name for name in person_data_names]
        with mp.Pool(processes=mp.cpu_count()) as p:
            person_data = list(tqdm(p.imap(read_person_data_csv, person_data_names),
                                    total=len(person_data_names)))
        train_data.extend(person_data)
    train = [train_index, train_data, train_label]

    file_processor = LoadSave()
    file_processor.save_data(path=".//data_tmp//train.pkl", data=train)


def processing_valid_data():
    def prefix_zero_padding(x):
        if len(x) == 1:
            return "00" + x
        elif len(x) == 2:
            return "0" + x
        return x

    PATH = ".//data//Validation Set//data//"
    person_data_names = os.listdir(PATH)
    person_data_names = sorted(person_data_names, key=lambda x: int(x[:-4]))

    valid_data, valid_index = [], []
    for name in person_data_names:
        valid_data.append(read_person_data_csv(PATH + name))
        valid_index.append(name[:-4])

    valid_index = list(map(prefix_zero_padding, valid_index))
    valid = [valid_index, valid_data]

    file_processor = LoadSave()
    file_processor.save_data(path=".//data_tmp//valid.pkl", data=valid)


if __name__ == "__main__":
    processing_train_data()
    # processing_valid_data()
