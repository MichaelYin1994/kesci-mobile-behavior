#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202009082036
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
利用钉钉（DingTalk）的API构建的训练监控机器人。
'''

import json
import urllib.request
from datetime import datetime

import tensorflow.keras as keras
import numpy as np


def send_msg_to_dingtalk(info_text, is_send_msg=False, is_print_msg=True):
    '''发送info_text给指定API_URL的钉钉机器人。'''
    if is_send_msg:
        API_URL = 'https://oapi.dingtalk.com/robot/send?access_token=d1b2a29b2ae62bc709693c02921ed097c621bc33e5963e9e0a5d5adf5eac10c1'

        # HTTP Head信息
        header = {
            'Content-Type': 'application/json',
            'Charset': 'UTF-8' }

        # 组装为json
        my_data = {
            'msgtype': 'markdown',
            'markdown': {'title': '[INFO]Neural Network at: {}'.format(datetime.now()),
                         'text': info_text},
            'at': {'isAtAll': False}}

        # 发送消息
        data_send = json.dumps(my_data)
        data_send = data_send.encode('utf-8')

        try:
            request = urllib.request.Request(url=API_URL, data=data_send, headers=header)
            opener = urllib.request.urlopen(request)
            opener.read()
        except:
            # 若无网络链接，则不执行操作
            pass

    if is_print_msg:
        print(info_text)


class RemoteMonitorDingTalk(keras.callbacks.Callback):
    '''
    在神经网络每一个batch训练结束之后，发送train和validation的信息给远程钉钉服务器。

    @Attributes:
    ----------
    is_send_msg: {bool-like}
        是否发送信息到远程服务器，若False，信息不进行发送。
    model_name: {str-like}
        该次训练的模型的名字。
    gpu_id: {int-like}
        当前模型使用的GPU的ID编号。
    '''
    def __init__(self, is_send_msg=False, model_name=None, gpu_id=0):
        super(RemoteMonitorDingTalk, self).__init__()
        self.is_send_msg = is_send_msg
        self.model_name = model_name
        self.gpu_id = gpu_id

    def on_epoch_end(self, epoch, logs):
        '''在每一个epoch之后，发送logs信息到远程服务器。'''
        log_keys = list(logs.keys())
        for k in log_keys:
            logs[k] = np.round(logs[k], 5)

        info_text = str(logs)
        if self.model_name is None:
            info_text = '[INFO] Epoch: {}, '.format(epoch) + info_text
        else:
            info_text = '[INFO][{}][GPU: {}] Epoch: {} '.format(
                self.model_name, self.gpu_id, epoch) + info_text

        send_msg_to_dingtalk(
            info_text, is_send_msg=self.is_send_msg, is_print_msg=False)
