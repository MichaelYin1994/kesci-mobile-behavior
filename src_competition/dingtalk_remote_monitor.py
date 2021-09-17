#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:50:59 2020

@author: zhuoyin94
"""

import numpy as np
import tensorflow.keras as keras
import json
from datetime import datetime
import urllib.request

def send_msg_to_dingtalk(info_text, is_send_msg=False, is_print_msg=True):
    """Msg sending block for the DingTalk bot."""
    if is_send_msg:
        API_URL = "https://oapi.dingtalk.com/robot/send?access_token=d1b2a29b2ae62bc709693c02921ed097c621bc33e5963e9e0a5d5adf5eac10c1"

        # HTTP header
        header = {
            "Content-Type": "application/json",
            "Charset": "UTF-8" }

        # DingTalk msg json
        my_data = {
            "msgtype": "markdown",
            "markdown": {"title": "[INFO]Neural Network at: {}".format(datetime.now()),
                         "text": info_text},
            "at": {"isAtAll": False}}

        # Sending the info_text
        data_send = json.dumps(my_data)
        data_send = data_send.encode("utf-8")

        try:
            request = urllib.request.Request(url=API_URL, data=data_send, headers=header)
            opener = urllib.request.urlopen(request)
            opener.read()
        except:
            # No network connection
            pass

    if is_print_msg:
        print(info_text)


class RemoteMonitorDingTalk(keras.callbacks.Callback):
    """
    Requirements: datetime.datetime, json, urllib.request
    """
    def __init__(self, is_send_msg=False):
        super(keras.callbacks.Callback, self).__init__()
        self.is_send_msg = is_send_msg

    def on_epoch_end(self, epoch, logs):
        log_keys = list(logs.keys())
        for k in log_keys:
            logs[k] = np.round(logs[k], 5)

        info_text = str(logs)
        info_text = "[INFO]Epoch: {}, ".format(epoch) + info_text
        send_msg_to_dingtalk(info_text, is_send_msg=self.is_send_msg,
                             is_print_msg=False)
