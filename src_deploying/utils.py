#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109171758
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
Helper functions.
'''

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import numba
from numba import njit
from tqdm import tqdm

from sklearn.metrics import auc, precision_recall_curve

warnings.filterwarnings('ignore')

def custom_eval_metric(y_true, y_pred):
    '''Custom evaluate metric.'''
    score_mapping = {
        0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'
    }

    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    pred_score = 0.0
    for i in range(len(y_true_idx)):
        code_y_true = score_mapping[y_true_idx[i]]
        code_y_pred = score_mapping[y_pred_idx[i]]

        if code_y_true == code_y_pred:
            pred_score += 1.0
        elif code_y_true.split("_")[0] == code_y_pred.split("_")[0]:
            pred_score += 1 / 7.0
        elif code_y_true.split("_")[1] == code_y_pred.split("_")[1]:
            pred_score += 1 / 3.0
        else:
            pred_score += 0.0

    avg_score = pred_score / len(y_true_idx)
    return avg_score


def tf_custom_eval(y_true, y_pred):
    '''Warp the python evaluate function.'''
    return tf.py_function(custom_eval_metric, (y_true, y_pred), tf.double)


@njit
def njit_f1(y_true_label, y_pred_label):
    '''计算F1分数，使用njit加速计算'''
    # https://www.itread01.com/content/1544007604.html
    tp = np.sum(np.logical_and(np.equal(y_true_label, 1),
                               np.equal(y_pred_label, 1)))
    fp = np.sum(np.logical_and(np.equal(y_true_label, 0),
                               np.equal(y_pred_label, 1)))
    # tn = np.sum(np.logical_and(np.equal(y_true, 1),
    #                            np.equal(y_pred_label, 0)))
    fn = np.sum(np.logical_and(np.equal(y_true_label, 1),
                               np.equal(y_pred_label, 0)))

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class LiteModel:
    '''将模型转换为Tensorflow Lite模型，提升推理速度。目前仅支持Keras模型转换。

    @Attributes:
    ----------
    interpreter: {Tensorflow lite transformed object}
        利用tf.lite.interpreter转换后的Keras模型。

    @References:
    ----------
    [1] https://medium.com/@micwurm/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
    '''

    @classmethod
    def from_file(cls, model_path):
        '''类方法。用于model_path下的模型，一般为*.h5模型。'''
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        '''类方法。用于直接转换keras模型。不用实例化类可直接调用该方法，返回
        被转换为tf.lite形式的Keras模型。

        @Attributes:
        ----------
        kmodel: {tf.keras model}
            待转换的Keras模型。

        @Returens:
        ----------
        经过转换的Keras模型。
        '''
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        '''为经过tf.lite.interpreter转换的模型构建构造输入输出的关键参数。

        TODO(zhuoyin94@163.com):
        ----------
        [1] 可添加关键字，指定converter选择采用INT8量化还是混合精度量化。
        [2] 可添加关键字，指定converter选择量化的方式：低延迟还是高推理速度？
        '''
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det['index']
        self.output_index = output_det['index']
        self.input_shape = input_det['shape']
        self.output_shape = output_det['shape']
        self.input_dtype = input_det['dtype']
        self.output_dtype = output_det['dtype']

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        ''' Like predict(), but only for a single record. The input data can be a Python list. '''
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
