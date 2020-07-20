#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:19:33 2020

@author: zhuoyin94
"""

import gc
import warnings
import multiprocessing as mp
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from utils import LoadSave, acc_combo, clf_pred_to_submission
from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk
###############################################################################
def load_data(name=None):
    """Load data from .//data_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//data_tmp//" + name)
    return data

if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")