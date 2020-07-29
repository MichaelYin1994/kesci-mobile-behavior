#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 04:39:26 2020

@author: gv1001107
"""


import pandas as pd
from sklearn.metrics import accuracy_score

sub_split = pd.read_csv(".//submissions//32_nn_split_15_vf1_7812_vacc_792_vc_8214.csv")
sub_normal = pd.read_csv(".//submissions//24_nn_15_vf1_7685_vacc_781_vc_8124.csv")

print("similarity: {:.5f}".format(
    accuracy_score(sub_split["behavior_id"].values.reshape((-1, 1)),
                   sub_normal["behavior_id"].values.reshape((-1, 1)))))