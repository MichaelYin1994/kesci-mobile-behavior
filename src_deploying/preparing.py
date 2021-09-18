#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109171549
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
Separate the raw training data into 2 parts:
- Model training part
- Inference part
'''

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

GLOBAL_RANDOM_SEED = 2912
np.random.seed(GLOBAL_RANDOM_SEED)


def save_to_csv(pack):
    '''Save a pandas DataFrame as a *.csv file.'''
    full_path_name, df = pack
    full_path_name = '{}_label_{}.csv'.format(
        full_path_name, int(df['behavior_id'].values[0])
    )

    df.to_csv(
        full_path_name, index=False
    )


if __name__ == "__main__":
    # Global parameters
    # **********************
    RATIO_FOR_TRAIN = 0.8
    DIR_NAME = '../data_tmp'

    # Preprocessing
    # **********************

    # Prepare the dir
    # --------
    if 'train_separated_csv' in os.listdir(DIR_NAME):
        fold_path = os.path.join(
            DIR_NAME, 'train_separated_csv'
        )
        for file_name in os.listdir(fold_path):
            os.remove(
                os.path.join(fold_path, file_name)
            )
    else:
        os.mkdir(
            os.path.join(DIR_NAME, 'train_separated_csv')
        )

    if 'test_separated_csv' in os.listdir(DIR_NAME):
        fold_path = os.path.join(
            DIR_NAME, 'test_separated_csv'
        )
        for file_name in os.listdir(fold_path):
            os.remove(
                os.path.join(fold_path, file_name)
            )
    else:
        os.mkdir(
            os.path.join(DIR_NAME, 'test_separated_csv')
        )

    # Split the data
    # --------
    train_df = pd.read_csv(
        os.path.join('../data', 'sensor_train.csv')
    )
    unqiue_ts_ids = train_df['fragment_id'].unique()

    # Shuffle the index
    # --------
    np.random.shuffle(unqiue_ts_ids)
    train_ids = [
        i for i in unqiue_ts_ids[:int(len(unqiue_ts_ids) * RATIO_FOR_TRAIN)]
    ]
    test_ids = [
        i for i in unqiue_ts_ids[int(len(unqiue_ts_ids) * RATIO_FOR_TRAIN):]
    ]

    # Gather the data
    # --------
    train_data_list = []
    for i in tqdm(train_ids):
        tmp_data = train_df.query(
            'fragment_id == {}'.format(i)
        )
        tmp_data.reset_index(drop=True, inplace=True)
        train_data_list.append(tmp_data)

    test_data_list = []
    for i in tqdm(test_ids):
        tmp_data = train_df.query(
            'fragment_id == {}'.format(i)
        )
        tmp_data.reset_index(drop=True, inplace=True)
        test_data_list.append(tmp_data)

    # Save to the local dir
    # **********************

    # Save training data
    # --------
    train_pack_list = [
        [os.path.join(
            DIR_NAME, 'train_separated_csv', 'ts_{}'.format(int(f_id))
        ), train_data_list[i]
        ] for i, f_id in enumerate(train_ids)
    ]
    for item in tqdm(train_pack_list):
        save_to_csv(item)

    # Save testing data
    # --------
    test_pack_list = [
        [os.path.join(
            DIR_NAME, 'test_separated_csv', 'ts_{}'.format(int(f_id))
        ), test_data_list[i]
        ] for i, f_id in enumerate(test_ids)
    ]
    for item in tqdm(test_pack_list):
        save_to_csv(item)
