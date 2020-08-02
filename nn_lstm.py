#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:15:12 2020

@author: gv1001107
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

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, Conv2D
from tensorflow.keras.layers import Input, concatenate, Add, ReLU, Flatten
from tensorflow.keras.layers import  GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling1D, AveragePooling1D
from tensorflow.keras import regularizers, constraints, optimizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from scipy.signal import resample

from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from utils import LoadSave, acc_combo, clf_pred_to_submission, plot_metric
from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk

# np.random.seed(2022)
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
###############################################################################
def load_data(name=None):
    """Load data from .//data_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//data_tmp//" + name)
    return data


def interp_seq(seq=None, length_interp=61):
    """Interpolating a seq to the fixed length_interp."""
    if len(seq) == length_interp:
        return seq
    interp_df = np.empty((length_interp, seq.shape[1]))
    interp_df[:] = np.nan

    # n_interps = length_interp - len(seq)
    # interp_pos = np.random.randint(1, len(seq)-1, n_interps)
    # interp_df = pd.DataFrame(interp_df, columns=list(seq.columns), index=interp_pos)
    # seq = seq.append(interp_df).sort_index()
    # seq = seq.interpolate(method="polynomial", order=3).reset_index(drop=True)

    for i in range(seq.shape[1]):
        interp_df[:, i] = resample(seq.values[:, i], length_interp)
    return interp_df


def split_seq(seq=None, strides=5, segment_length=20, padding=None):
    """Split the time serie seq according to the strides and segment_length."""
    # if len(seq) < (segment_length + strides):
    #     raise ValueError("The length of seq is less than the segment_length + strides !")
    if padding is not None and padding not in ["zero", "backward"]:
        raise ValueError("Invalid padding method !")

    # Split the time series seq
    seq_split = []
    split_pos = [i for i in list(range(0, len(seq), strides)) if i + segment_length <= len(seq)]

    for pos in split_pos:
        seq_split.append(seq[pos:(pos+segment_length), :])

    # Processing the remain unsplit segments
    if padding is None:
        pass
    elif padding == "backward":
        seq_split.append(seq[(len(seq)-segment_length):, :])
    else:
        seq_tmp = seq[(split_pos[-1]+segment_length):, :]
        n_need_to_pad = segment_length - len(seq_tmp)
        seq_split.append(np.vstack((seq_tmp, np.zeros((n_need_to_pad, seq.shape[1])))))
    return seq_split


def preprocessing_seq(seq=None, length_interp=63, **kwargs):
    """Interpolating a seq on selected feattures to the fixed length_interp"""
    seq["mod"] = np.sqrt(seq["acc_x"]**2 + seq["acc_y"]**2 + seq["acc_z"]**2)
    seq["modg"] = np.sqrt(seq["acc_xg"]**2 + seq["acc_yg"]**2 + seq["acc_zg"]**2)

    selected_feats = ["acc_x", "acc_y", "acc_z", "acc_xg", "acc_yg", "acc_zg", "mod", "modg"]
    seq_val = interp_seq(seq[selected_feats], length_interp=length_interp)
    seq_val = split_seq(seq_val, **kwargs)

    labels = [np.nan] * len(seq_val)
    if "behavior_id" in seq.columns:
        labels = [seq["behavior_id"].iloc[0]] * len(seq_val)
    id_names = [seq["fragment_id"].iloc[0]] * len(seq_val)

    return seq_val, labels, id_names


def build_model(verbose=False, is_compile=True, **kwargs):
    series_length = kwargs.pop("series_length", 61)
    series_feat_size = kwargs.pop("series_feat_size", 8)
    layer_input_series = Input(shape=(series_length, series_feat_size), name="input_series")
    layer_input_acc = Input(shape=(series_length,), dtype='int32', name="input_creative_id")
    layer_input_pos = Input(shape=(series_length,), dtype='int32', name="input_creative_id")


    # CONV_2d cross channel
    # -----------------
    



    return None


class GensimCallback(CallbackAny2Vec):
    """Callback Class for monintering the w2v training status.
    Reference: https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    """
    def __init__(self):
        self.epoch = 0
        self.loss = [0]

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_decreasing_precent = (loss - self.loss[-1])/loss * 100
        print("-- Loss after epoch {}: {:.2f}, decreasing {:.5f}%.".format(
            self.epoch, loss, loss_decreasing_precent))
        self.epoch += 1
        self.loss.append(loss)


def compute_cbow_embedding(corpus=None,
                           embedding_size=10,
                           iters=40, 
                           min_count=4,
                           window_size=3, 
                           seed=2020,
                           is_save_model=True,
                           model_name="cbow_model"):
    """Using CBOW to train the text embedding model.
    corpus example:
        [["1", "2", "3"],
        ...,
        ["10", "23", "65", "9", "34"]]
    """
    print("\n[INFO]Start CBOW word embedding at {}".format(datetime.now()))
    print("-------------------------------------------")
    model = word2vec.Word2Vec(corpus,
                              size=embedding_size,
                              min_count=min_count,
                              workers=mp.cpu_count(),
                              window=window_size,
                              compute_loss=True,
                              callbacks=[GensimCallback()],
                              seed=seed,
                              iter=iters,
                              sg=0)
    print("-------------------------------------------")
    print("[INFO]End CBOW word embedding at {}\n".format(datetime.now()))

    # Save the embedding model.
    file_processor = LoadSave()
    if is_save_model:
        file_processor.save_data(path=".//models//{}.pkl".format(model_name),
                                  data=model)
    return model


def compute_skip_gram_embedding(corpus=None,
                                embedding_size=10, 
                                iters=40,
                                min_count=4,
                                window_size=3,
                                seed=2020, 
                                is_save_model=True,
                                model_name="skip_gram_model"):
    """Using Skip-Gram to train the text embedding model.
    corpus example:
        [["1", "2", "3"],
        ...,
        ["10", "23", "65", "9", "34"]]
    """
    print("\n[INFO]Start Skip-Gram word embedding at {}".format(datetime.now()))
    print("-------------------------------------------")
    model = word2vec.Word2Vec(corpus,
                              size=embedding_size,
                              min_count=min_count,
                              workers=mp.cpu_count(),
                              window=window_size,
                              compute_loss=True,
                              callbacks=[GensimCallback()],
                              seed=seed,
                              iter=iters,
                              sg=1)
    print("-------------------------------------------")
    print("[INFO]End Skip-Gram word embedding at {}\n".format(datetime.now()))

    # Save the embedding model.
    file_processor = LoadSave()
    if is_save_model:
        file_processor.save_data(path=".//models//{}.pkl".format(model_name),
                                  data=model)
    return model


def seq_pos_to_corpus(seq=None):
    """Tranform a sequence into a corpus."""
    # For numeric stable
    # seq["acc_x"][seq["acc_x"].between(-0.00001, 0.00001)] = 0
    # seq["acc_y"][seq["acc_y"].between(-0.00001, 0.00001)] = 0
    # seq["acc_z"][seq["acc_z"].between(-0.00001, 0.00001)] = 0

    seq["acc_x"] = np.round(seq["acc_x"].values, 1)
    seq["acc_y"] = np.round(seq["acc_y"].values, 1)
    seq["acc_z"] = np.round(seq["acc_z"].values, 1)

    corpus = seq["acc_x"].apply(str) + "_" + seq["acc_y"].apply(str) + "_" + seq["acc_z"].apply(str)
    corpus = corpus.values.tolist()
    return corpus


def seq_acc_to_corpus(seq=None):
    """Transform the acc info to corpus."""
    # For numeric stable
    # seq["acc_xg"][seq["acc_xg"].between(-0.00001, 0.00001)] = 0
    # seq["acc_yg"][seq["acc_yg"].between(-0.00001, 0.00001)] = 0
    # seq["acc_zg"][seq["acc_zg"].between(-0.00001, 0.00001)] = 0

    seq["acc_xg"] = np.round(seq["acc_xg"].values, 1)
    seq["acc_yg"] = np.round(seq["acc_yg"].values, 1)
    seq["acc_zg"] = np.round(seq["acc_zg"].values, 1)

    corpus = seq["acc_xg"].apply(str) + "_" + seq["acc_yg"].apply(str) + "_" + seq["acc_zg"].apply(str)
    corpus = corpus.values.tolist()
    return corpus


def corpus_to_sequence(corpus=None):
    # New corpus
    word_index, new_corpus, count = {}, [], 0
    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            if word in word_index:
                new_sentence.append(word_index[word])
            else:
                new_sentence.append(str(count))
                word_index[word] = str(count)
                count += 1
        new_corpus.append(new_sentence)
    return new_corpus, word_index


# def build_embedding_matrix(word_index=None, embedding_index=None,
#                            max_feats=300, embedding_size=100,
#                            verbose=True):
#     """Mapping words in word_index into word vectors.
#     Refs:
#     [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
#     [2] https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
#     """
#     embedding_mat = np.zeros((max_feats, embedding_size))
#     is_contain_nan = "nan" in embedding_index

#     for word, i in tqdm(word_index.items(), disable=not verbose):
#         if i >= max_feats:
#             continue
#         try:
#             embedding_vec = embedding_index[word]
#         except:
#             if is_contain_nan:
#                 embedding_vec = embedding_index["nan"]
#             else:
#                 embedding_vec = [0] * embedding_size
#         if embedding_vec is not None:
#             embedding_mat[i] = embedding_vec
#     return embedding_mat


if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")

    total_data = train_data + test_data
    fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    seq = total_data[14]

    total_feats = pd.DataFrame(None)
    total_feats["fragment_id"] = fragment_id
    total_feats["behavior_id"] = labels + [np.nan] * len(test_data)
    total_feats["is_train"] = [True] * len(train_data) + [False] * len(test_data)

    SENDING_TRAINING_INFO = False
    send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)
    INFO_TEXT = "[BEGIN]#Training: {}, #Testing: {}, at: {}".format(
        len(train_data),
        len(test_data),
        str(datetime.now())[:-7])
    send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    ##########################################################################
    # Step 1: Interpolate all the sequence to the fixed length
    # ------------------------
    res = preprocessing_seq(total_data[0].copy())
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(preprocessing_seq, total_data),
                        total=len(total_data)))

    total_split_data, seq_id, seq_label = [], [], []
    for item in tmp:
        total_split_data.extend(item[0])
        seq_label.extend(item[1])
        seq_id.extend(item[2])

    total_split_feats = pd.DataFrame(None)
    total_split_feats["behavior_id"] = seq_label
    total_split_feats["fragment_id"] = seq_id
    n_split_train, n_split_test = total_split_feats["behavior_id"].notnull().sum(), total_split_feats["behavior_id"].isnull().sum()

    INFO_TEXT = "[INFO] #splite train: {}, #split test: {}, segment_shape: {}".format(
        n_split_train, n_split_test, len(tmp[0][0]))
    send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    train_seq, test_seq = total_split_data[:n_split_train], total_split_data[n_split_train:]
    train_seq, test_seq = np.array(train_seq), np.array(test_seq)

    train_group_id = total_split_feats[total_split_feats["behavior_id"].notnull()]["fragment_id"].values
    test_group_id = total_split_feats[total_split_feats["behavior_id"].isnull()]["fragment_id"].values
    split_labels = total_split_feats[total_split_feats["behavior_id"].notnull()]["behavior_id"].values

    ##########################################################################
    # Step 2: Position and position-with-acc to corpus
    # ------------------------
    total_split_data = [pd.DataFrame(item, columns=["acc_x", "acc_y", "acc_z",
                                                    "acc_xg", "acc_yg", "acc_zg",
                                                    "mod", "modg"]) for item in total_split_data]

    # Preprocessing of pos
    # ------------------------
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(seq_pos_to_corpus, total_split_data),
                        total=len(total_split_data)))
    corpus_pos, _ = corpus_to_sequence(tmp)

    vocab_max_feats_pos = 60000
    tokenizer = Tokenizer(num_words=vocab_max_feats_pos, oov_token="nan")
    tokenizer.fit_on_texts(corpus_pos)
    vocab_pos = tokenizer.word_index
    corpus_pos = np.array(tokenizer.texts_to_sequences(corpus_pos))
    train_corpus_pos, test_corpus_pos = corpus_pos[:n_split_train], corpus_pos[n_split_train:]

    # Preprocessing of pos with acc
    # ------------------------
    res = seq_acc_to_corpus(seq)
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(seq_acc_to_corpus, total_split_data),
                        total=len(total_split_data)))
    corpus_acc, _ = corpus_to_sequence(tmp)

    vocab_max_feats_acc = 200000
    tokenizer = Tokenizer(num_words=vocab_max_feats_acc, oov_token="nan")
    tokenizer.fit_on_texts(corpus_acc)
    vocab_acc = tokenizer.word_index
    corpus_acc = np.array(tokenizer.texts_to_sequences(corpus_acc))
    train_corpus_acc, test_corpus_acc = corpus_acc[:n_split_train], corpus_acc[n_split_train:]

    # ##########################################################################
    # # Step 3: Embedding the corpus
    # # ------------------------
    # model_cbow_pos = compute_cbow_embedding(corpus=corpus_pos,
    #                                         embedding_size=40,
    #                                         window_size=3,
    #                                         min_count=4,
    #                                         iters=20,
    #                                         is_save_model=False,
    #                                         model_name="cbow_pos")
    # model_cbow_acc = compute_cbow_embedding(corpus=corpus_acc,
    #                                         embedding_size=40,
    #                                         window_size=3,
    #                                         min_count=4,
    #                                         iters=20,
    #                                         is_save_model=False,
    #                                         model_name="cbow_acc")

    # # # model_sg_pos = compute_skip_gram_embedding(corpus=corpus_pos,
    # # #                                             embedding_size=40,
    # # #                                             window_size=3,
    # # #                                             min_count=4,
    # # #                                             iters=20,
    # # #                                             is_save_model=False,
    # # #                                             model_name="sg_pos")
    # # # model_sg_acc = compute_skip_gram_embedding(corpus=corpus_acc,
    # # #                                             embedding_size=3,
    # # #                                             window_size=3,
    # # #                                             min_count=4,
    # # #                                             iters=20,
    # # #                                             is_save_model=False,
    # # #                                             model_name="sg_acc")
    # model_pos, model_acc = model_cbow_pos, model_cbow_acc

    ##########################################################################
    # # Step 4: Build the embedding matrix
    # # ------------------------
    # embedding_mat_pos = build_embedding_matrix(word_index=vocab_pos,
    #                                            embedding_index=model_pos,
    #                                            max_feats=vocab_max_feats_pos)
