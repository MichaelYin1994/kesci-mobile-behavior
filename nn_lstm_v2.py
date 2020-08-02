#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 03:11:38 2020

@author: gv1001107
"""


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
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling1D, AveragePooling1D, SpatialDropout1D
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
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

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
    seq["acc_x"] = np.round(seq["acc_x"].values, 1)
    seq["acc_y"] = np.round(seq["acc_y"].values, 1)
    seq["acc_z"] = np.round(seq["acc_z"].values, 1)

    corpus = seq["acc_x"].apply(str) + "_" + seq["acc_y"].apply(str) + "_" + seq["acc_z"].apply(str)
    corpus = corpus.values.tolist()
    return corpus


def seq_acc_to_corpus(seq=None):
    """Transform the acc info to corpus."""
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


def build_embedding_matrix(word_index=None, embedding_index=None,
                           max_feats=300, embedding_size=100,
                           verbose=True):
    """Mapping words in word_index into word vectors.
    Refs:
    [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    [2] https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
    """
    embedding_mat = np.zeros((max_feats, embedding_size))
    is_contain_nan = "nan" in embedding_index

    for word, i in tqdm(word_index.items(), disable=not verbose):
        if i >= max_feats:
            continue
        try:
            embedding_vec = embedding_index[word]
        except:
            if is_contain_nan:
                embedding_vec = embedding_index["nan"]
            else:
                embedding_vec = [0] * embedding_size
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    return embedding_mat



def build_model(verbose=False, is_compile=True, **kwargs):
    series_length = kwargs.pop("series_length", 61)
    series_feat_size = kwargs.pop("series_feat_size", 8)
    layer_input_series = Input(shape=(series_length,
                                      series_feat_size),
                               name="input_series")

    # Create embedding
    # -----------------
    layer_input_pos = Input(shape=(series_length, ), dtype='int32', 
                            name="input_pos_id")
    vocab_size_pos = kwargs.pop("vocab_size_pos", 1000)
    embedding_size_pos = kwargs.pop("embedding_size_pos", 40)
    embedding_mat_pos = kwargs.pop("embedding_mat_pos", None)
    embedding_pos = Embedding(vocab_size_pos,
                              embedding_size_pos,
                              input_length=series_length,
                              weights=[embedding_mat_pos],
                              name="embedding_pos",
                              trainable=False)(layer_input_pos)


    layer_input_acc = Input(shape=(series_length, ), dtype='int32', 
                            name="input_acc_id")
    vocab_size_acc = kwargs.pop("vocab_size_acc", 1000)
    embedding_size_acc = kwargs.pop("embedding_size_acc", 40)
    embedding_mat_acc = kwargs.pop("embedding_mat_acc", None)
    embedding_acc = Embedding(vocab_size_acc,
                              embedding_size_acc,
                              input_length=series_length,
                              weights=[embedding_mat_acc],
                              name="embedding_acc",
                              trainable=False)(layer_input_acc)

    # concat all input
    # -----------------
    layer_total = concatenate([embedding_acc, embedding_pos, layer_input_series])

    layer_total = SpatialDropout1D(0.15)(layer_total)
    layer_lstm_0 = Bidirectional(GRU(40, return_sequences=True))(layer_total)
    layer_lstm_0 = SpatialDropout1D(0.15)(layer_lstm_0)
    layer_lstm_1 = Bidirectional(GRU(40, return_sequences=True))(layer_lstm_0)

    # layer_lstm_0 = AveragePooling
    layer_avg_pool = GlobalAveragePooling1D()(layer_lstm_1)

    # Concat all
    # -----------------
    # layer_pooling = concatenate([layer_avg_pool])

    # Output structure
    # -----------------
    layer_output = Dropout(0.2)(layer_avg_pool)
    layer_output = Dense(128, activation="relu")(layer_output)
    layer_output = Dense(19, activation='softmax')(layer_output)

    model = Model([layer_input_series, layer_input_pos, layer_input_acc], layer_output)
    if verbose:
        model.summary()
    if is_compile:
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(0.003, decay=1e-6), metrics=['acc'])
    return model


if __name__ == "__main__":
    # train_data = load_data("train.pkl")
    # test_data = load_data("test.pkl")

    # total_data = train_data + test_data
    # labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    # seq = total_data[14]

    # total_feats = pd.DataFrame(None)
    # total_feats["fragment_id"] = [seq["fragment_id"].unique()[0] for seq in total_data]
    # total_feats["behavior_id"] = labels + [np.nan] * len(test_data)
    # total_feats["is_train"] = [True] * len(train_data) + [False] * len(test_data)

    # SENDING_TRAINING_INFO = False
    # send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)
    # INFO_TEXT = "[BEGIN]#Training: {}, #Testing: {}, at: {}".format(
    #     len(train_data),
    #     len(test_data),
    #     str(datetime.now())[:-7])
    # send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    # ##########################################################################
    # # Step 1: Interpolate all the sequence to the fixed length
    # # ------------------------
    # res = preprocessing_seq(total_data[0].copy())
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(preprocessing_seq, total_data),
    #                     total=len(total_data)))

    # total_split_data, seq_id, seq_label = [], [], []
    # for item in tmp:
    #     total_split_data.extend(item[0])
    #     seq_label.extend(item[1])
    #     seq_id.extend(item[2])

    # total_split_feats = pd.DataFrame(None)
    # total_split_feats["behavior_id"] = seq_label
    # total_split_feats["fragment_id"] = seq_id
    # n_split_train, n_split_test = total_split_feats["behavior_id"].notnull().sum(), total_split_feats["behavior_id"].isnull().sum()

    # INFO_TEXT = "[INFO] #splite train: {}, #split test: {}, segment_shape: {}".format(
    #     n_split_train, n_split_test, len(tmp[0][0]))
    # send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    # train_seq, test_seq = total_split_data[:n_split_train], total_split_data[n_split_train:]
    # train_seq, test_seq = np.array(train_seq), np.array(test_seq)

    # train_group_id = total_split_feats[total_split_feats["behavior_id"].notnull()]["fragment_id"].values
    # test_group_id = total_split_feats[total_split_feats["behavior_id"].isnull()]["fragment_id"].values
    # split_labels = total_split_feats[total_split_feats["behavior_id"].notnull()]["behavior_id"].values

    # ##########################################################################
    # # Step 2: Position and position-with-acc to corpus
    # # ------------------------
    # total_split_data = [pd.DataFrame(item, columns=["acc_x", "acc_y", "acc_z",
    #                                                 "acc_xg", "acc_yg", "acc_zg",
    #                                                 "mod", "modg"]) for item in total_split_data]

    # # Preprocessing of pos
    # # ------------------------
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(seq_pos_to_corpus, total_split_data),
    #                     total=len(total_split_data)))
    # corpus_pos, _ = corpus_to_sequence(tmp)

    # vocab_max_feats_pos = 50000
    # tokenizer = Tokenizer(num_words=vocab_max_feats_pos, oov_token="nan")
    # tokenizer.fit_on_texts(corpus_pos)
    # vocab_pos = tokenizer.word_index
    # corpus_pos_transfer = np.array(tokenizer.texts_to_sequences(corpus_pos))
    # train_id_pos, test_id_pos = corpus_pos_transfer[:n_split_train], corpus_pos_transfer[n_split_train:]

    # # Preprocessing of pos with acc
    # # ------------------------
    # res = seq_acc_to_corpus(seq)
    # with mp.Pool(processes=mp.cpu_count()) as p:
    #     tmp = list(tqdm(p.imap(seq_acc_to_corpus, total_split_data),
    #                     total=len(total_split_data)))
    # corpus_acc, _ = corpus_to_sequence(tmp)

    # vocab_max_feats_acc = 150000
    # tokenizer = Tokenizer(num_words=vocab_max_feats_acc, oov_token="nan")
    # tokenizer.fit_on_texts(corpus_acc)
    # vocab_acc = tokenizer.word_index
    # corpus_acc_transfer = np.array(tokenizer.texts_to_sequences(corpus_acc))
    # train_id_acc, test_id_acc = corpus_acc_transfer[:n_split_train], corpus_pos_transfer[n_split_train:]

    # ##########################################################################
    # # Step 3: Training word2vec embedding
    # # ------------------------
    # model_cbow_pos = compute_cbow_embedding(corpus=corpus_pos,
    #                                         embedding_size=40,
    #                                         window_size=5,
    #                                         min_count=4,
    #                                         iters=20,
    #                                         is_save_model=False,
    #                                         model_name="cbow_pos")
    # model_cbow_acc = compute_cbow_embedding(corpus=corpus_acc,
    #                                         embedding_size=40,
    #                                         window_size=5,
    #                                         min_count=4,
    #                                         iters=20,
    #                                         is_save_model=False,
    #                                         model_name="cbow_acc")

    # # # model_sg_pos = compute_skip_gram_embedding(corpus=corpus_pos,
    # # #                                             embedding_size=40,
    # # #                                             window_size=5,
    # # #                                             min_count=4,
    # # #                                             iters=20,
    # # #                                             is_save_model=False,
    # # #                                             model_name="sg_pos")
    # # # model_sg_acc = compute_skip_gram_embedding(corpus=corpus_acc,
    # # #                                             embedding_size=40,
    # # #                                             window_size=5,
    # # #                                             min_count=4,
    # # #                                             iters=20,
    # # #                                             is_save_model=False,
    # # #                                             model_name="sg_acc")
    # model_pos, model_acc = model_cbow_pos, model_cbow_acc
    # ##########################################################################
    # # Step 4: Bulid the pre-training embedding matrix
    # # ------------------------
    # embedding_mat_pos = build_embedding_matrix(word_index=vocab_pos,
    #                                             embedding_index=model_pos,
    #                                             max_feats=vocab_max_feats_pos,
    #                                             embedding_size=40)

    # embedding_mat_acc = build_embedding_matrix(word_index=vocab_acc,
    #                                             embedding_index=model_acc,
    #                                             max_feats=vocab_max_feats_acc,
    #                                             embedding_size=40)

    ##########################################################################
    # Step 5: Training NN
    # ------------------------
    N_FOLDS = 5
    BATCH_SIZE = 2048
    N_EPOCHS = 700
    IS_STRATIFIED = False
    SEED = 2090
    PLOT_TRAINING = False

    folds = GroupKFold(n_splits=N_FOLDS)
    scores = np.zeros((N_FOLDS, 7))
    oof_pred = np.zeros((len(train_data), 19))
    y_pred = np.zeros((len(test_data), 19))
    early_stop = EarlyStopping(monitor='val_acc',
                               mode='max',
                               verbose=1,
                               patience=100,
                               restore_best_weights=True)

    # Training the NN classifier
    send_msg_to_dingtalk("\n[INFO]Start training NeuralNets CLASSIFIER at: {}".format(
        str(datetime.now())[:-7]), is_send_msg=SENDING_TRAINING_INFO)
    print("==================================")
    targets_oht = to_categorical(split_labels)
    for fold, (split_tra_id, split_val_id) in enumerate(folds.split(train_seq, targets_oht, train_group_id)):
        d_train, d_valid = train_seq[split_tra_id], train_seq[split_val_id]
        d_train_id_pos,  d_valid_id_pos = train_id_pos[split_tra_id], train_id_pos[split_val_id]
        d_train_id_acc,  d_valid_id_acc = train_id_acc[split_tra_id], train_id_acc[split_val_id]
        t_train, t_valid = targets_oht[split_tra_id], targets_oht[split_val_id]

        # Destroy all graph nodes in memory
        # ---------------
        K.clear_session()
        gc.collect()

        # Training NN classifier
        # ---------------
        model = build_model(verbose=False,
                            is_complie=True,
                            series_length=train_seq.shape[1],
                            series_feat_size=train_seq.shape[2],
                            
                            embedding_mat_pos=embedding_mat_pos,
                            embedding_size_pos=40,
                            vocab_size_pos=vocab_max_feats_pos,
                            
                            embedding_mat_acc=embedding_mat_acc,
                            embedding_size_acc=40,
                            vocab_size_acc=vocab_max_feats_acc)

        history = model.fit(x=[d_train, d_train_id_pos, d_train_id_acc],
                            y=t_train,
                            validation_data=([d_valid, d_valid_id_pos, d_valid_id_acc], t_valid),
                            callbacks=[early_stop],
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            verbose=2)

        # Model trianing plots
        # ---------------
        if PLOT_TRAINING:
            plot_metric(history, metric_type="acc")
            plt.savefig(".//plots//training_fold_acc_{}.png".format(fold),
                        bbox_inches="tight", dpi=500)

            plot_metric(history, metric_type="loss")
            plt.savefig(".//plots//training_fold_loss_{}.png".format(fold),
                        bbox_inches="tight", dpi=500)
            plt.close("all")

        # Training evaluation
        # ---------------
        train_pred_proba = model.predict(x=[d_train, d_train_id_pos, d_train_id_acc],
                                          batch_size=BATCH_SIZE)
        valid_pred_proba = model.predict(x=[d_valid, d_valid_id_pos, d_valid_id_acc],
                                          batch_size=BATCH_SIZE)
        y_pred_proba = model.predict(x=[test_seq, test_id_pos, test_id_acc],
                                     batch_size=BATCH_SIZE)

        # ---------------
        train_pred_proba = pd.DataFrame(train_pred_proba,
                                        index=train_group_id[split_tra_id])
        valid_pred_proba = pd.DataFrame(valid_pred_proba,
                                        index=train_group_id[split_val_id])
        y_pred_proba = pd.DataFrame(y_pred_proba,
                                    index=test_group_id)
        train_pred_proba = train_pred_proba.groupby(
            train_pred_proba.index).agg(np.mean).values
        valid_pred_proba = valid_pred_proba.groupby(
            valid_pred_proba.index).agg(np.mean).values
        y_pred_proba = y_pred_proba.groupby(
            y_pred_proba.index).agg(np.mean).values

        df_train = pd.DataFrame(None, index=train_group_id[split_tra_id])
        df_valid = pd.DataFrame(None, index=train_group_id[split_val_id])
        df_train["labels"], df_valid["labels"] = split_labels[split_tra_id], split_labels[split_val_id]

        # Training oof_pred and y_pred construction
        # ---------------
        y_pred += y_pred_proba / N_FOLDS
        oof_val_id = df_valid.groupby(df_valid.index).agg(pd.Series.mode).reset_index()["index"].values - 10000000
        oof_pred[oof_val_id] = valid_pred_proba

        # Training figures construction
        # ---------------
        train_pred_label = np.argmax(train_pred_proba, axis=1).reshape((-1, 1))
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        t_train_label = df_train.groupby(df_train.index).agg(pd.Series.mode).values.reshape((-1, 1))
        t_valid_label = df_valid.groupby(df_valid.index).agg(pd.Series.mode).values.reshape((-1, 1))

        train_f1 = f1_score(
            t_train_label, train_pred_label, average="macro")
        train_acc = accuracy_score(t_train_label,
                                    train_pred_label)
        valid_f1 = f1_score(
            t_valid_label, valid_pred_label, average="macro")
        valid_acc = accuracy_score(t_valid_label,
                                    valid_pred_label)

        train_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_train_label, train_pred_label))).mean()
        valid_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((t_valid_label, valid_pred_label))).mean()

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = train_f1, train_acc
        scores[fold, 3], scores[fold, 4] = valid_f1, valid_acc
        scores[fold, 5], scores[fold, 6] = train_custom, valid_custom

        INFO_TEXT = "[INFO] folds {}({}), valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
            fold+1, N_FOLDS, valid_f1, valid_acc, valid_custom)
        send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
    total_f1 = f1_score(np.array(labels).reshape(-1, 1),
                        oof_pred_label.reshape((-1, 1)), average="macro")
    total_acc = accuracy_score(np.array(labels).reshape(-1, 1),
                               oof_pred_label.reshape((-1, 1)))
    total_custom = np.apply_along_axis(
            acc_combo, 1, np.hstack((np.array(labels).reshape((-1, 1)),
                                     oof_pred_label.reshape((-1, 1))))).mean()

    INFO_TEXT = "[INFO] total valid f1: {:.5f}, acc: {:.5f}, custom: {:.5f}".format(
        total_f1, total_acc, total_custom)
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)
    INFO_TEXT = "[INFO]End training NeuralNets CLASSIFIER at: {}\n".format(
        str(datetime.now())[:-7])
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=SENDING_TRAINING_INFO)
    send_msg_to_dingtalk("++++++++++++++++++++++++++++", SENDING_TRAINING_INFO)

    # Saving prediction results
    # ------------------------
    scores = pd.DataFrame(scores, columns=["folds", "train_f1", "train_acc",
                                           "valid_f1", "valid_acc",
                                           "train_custom", "valid_custom"])
    y_pred = pd.DataFrame(
        y_pred, columns=["y_pred_{}".format(i) for i in range(19)])
    y_pred["fragment_id"] = total_feats.query("is_train == False")["fragment_id"].values
    oof_pred = pd.DataFrame(
        oof_pred, columns=["oof_pred_{}".format(i) for i in range(19)])
    oof_pred["fragment_id"], oof_pred["behavior_id"] = total_feats.query("is_train == True")["fragment_id"].values, total_feats.query("is_train == True")["behavior_id"].values

    clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
                           target_name="behavior_id", id_name="fragment_id",
                           sub_str_field="nn_split_{}".format(N_FOLDS), save_oof=True)
