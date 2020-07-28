#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:28:38 2020

@author: zhuoyin94
"""

import gc
import multiprocessing as mp
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from utils import LoadSave, lightgbm_classifier_training, clf_pred_to_submission

np.random.seed(9102)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def load_data(name=None):
    """Load data from .//data_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//data_tmp//" + name)
    return data


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

    # Save the emebdding model.
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

    # Save the emebdding model.
    file_processor = LoadSave()
    if is_save_model:
        file_processor.save_data(path=".//models//{}.pkl".format(model_name),
                                  data=model)
    return model


def compute_mean_embedding(corpus=None,
                           model=None,
                           prefix="cbow_industry"):
    """Sum the embedding vector of each word in the corpus."""
    embedding_vec = []
    embedding_size = model.layer1_size

    for sent in corpus:
        sent_vec, word_count = 0, 0
        for word in sent:
            if word in model:
                sent_vec += model[word]
                word_count += 1
            else:
                continue
        if word_count == 0:
            embedding_vec.append([0] * embedding_size)
        else:
            embedding_vec.append(sent_vec/word_count)
    embedding_vec = np.array(embedding_vec)
    embedding_df = pd.DataFrame(embedding_vec, columns=[
        "embedding_{}_{}".format(prefix, i) for i in range(embedding_size)])
    return embedding_df


def seq_pos_to_corpus(seq=None):
    """Tranform a sequence into a corpus."""
    # For numeric stable
    seq["acc_x"][seq["acc_x"].between(-0.00001, 0.00001)] = 0
    seq["acc_y"][seq["acc_y"].between(-0.00001, 0.00001)] = 0
    seq["acc_z"][seq["acc_z"].between(-0.00001, 0.00001)] = 0

    corpus = seq["acc_x"].apply(str) + "_" + seq["acc_y"].apply(str) + "_" + seq["acc_z"].apply(str)
    corpus = corpus.values.tolist()
    return corpus


def seq_acc_to_corpus(seq=None):
    """Transform the acc info to corpus."""
    # For numeric stable
    seq["acc_xg"][seq["acc_xg"].between(-0.00001, 0.00001)] = 0
    seq["acc_yg"][seq["acc_yg"].between(-0.00001, 0.00001)] = 0
    seq["acc_zg"][seq["acc_zg"].between(-0.00001, 0.00001)] = 0

    corpus = seq["acc_xg"].apply(str) + "_" + seq["acc_yg"].apply(str) + "_" + seq["acc_zg"].apply(str)
    corpus = corpus.values.tolist()
    return corpus


def compute_tfidf_feats(corpus=None, max_feats=100):
    """Total tf-idf features."""
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2",
                                 max_features=max_feats, analyzer='word',
                                 ngram_range=(1, 1),
                                 token_pattern=r"(?u)\b\w+\b")
    tfidf_feats = vectorizer.fit_transform(corpus).toarray()
    return tfidf_feats


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


if __name__ == "__main__":
    train_data = load_data("train.pkl")
    test_data = load_data("test.pkl")

    total_data = train_data + test_data
    fragment_id = [seq["fragment_id"].unique()[0] for seq in total_data]
    labels = [seq["behavior_id"].unique()[0] for seq in train_data]
    seq = total_data[14]

    # mapping = {0: 0, 1: 0, 2: 0, 3: 0,
    #             4: 3, 5: 0, 6: 1, 7: 1,
    #             8: 1, 9: 1, 10: 1, 11: 0,
    #             12: 2, 13: 2, 14: 2, 15: 2,
    #             16: 2, 17: 2, 18: 2}
    # labels = [mapping[i] for i in labels]

    total_feats = pd.DataFrame(None)
    total_feats["fragment_id"] = fragment_id
    total_feats["behavior_id"] = labels + [np.nan] * len(test_data)

    # Step 1: Creating corpus
    # ------------------------
    res = seq_pos_to_corpus(seq)
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(seq_pos_to_corpus, total_data),
                        total=len(total_data)))
    corpus_pos, word_index_pos = corpus_to_sequence(tmp)

    res = seq_acc_to_corpus(seq)
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(seq_acc_to_corpus, total_data),
                        total=len(total_data)))
    corpus_acc, word_index_acc = corpus_to_sequence(tmp)

    # Step 2: Embedding the corpus
    # ------------------------
    model_cbow_pos = compute_cbow_embedding(corpus=corpus_pos,
                                            embedding_size=40,
                                            window_size=3,
                                            min_count=4,
                                            iters=20,
                                            is_save_model=True,
                                            model_name="cbow_pos")
    model_cbow_acc = compute_cbow_embedding(corpus=corpus_acc,
                                            embedding_size=30,
                                            window_size=3,
                                            min_count=4,
                                            iters=20,
                                            is_save_model=True,
                                            model_name="cbow_acc")

    model_sg_pos = compute_skip_gram_embedding(corpus=corpus_pos,
                                                embedding_size=40,
                                                window_size=3,
                                                min_count=4,
                                                iters=20,
                                                is_save_model=True,
                                                model_name="sg_pos")
    model_sg_acc = compute_skip_gram_embedding(corpus=corpus_acc,
                                                embedding_size=30,
                                                window_size=3,
                                                min_count=4,
                                                iters=20,
                                                is_save_model=True,
                                                model_name="sg_acc")

    df_cbow_pos = compute_mean_embedding(corpus=corpus_pos,
                                          model=model_cbow_pos,
                                          prefix="cbow_pos")
    df_cbow_acc = compute_mean_embedding(corpus=corpus_acc,
                                          model=model_cbow_acc,
                                          prefix="cbow_acc")
    df_sg_pos = compute_mean_embedding(corpus=corpus_pos,
                                        model=model_sg_pos,
                                        prefix="sg_pos")
    df_sg_acc = compute_mean_embedding(corpus=corpus_acc,
                                        model=model_sg_acc,
                                        prefix="sg_acc")
    embedding_feats = pd.concat([df_cbow_pos, df_cbow_acc, df_sg_pos, df_sg_acc], axis=1)
    embedding_feats["fragment_id"] = fragment_id

    file_processor = LoadSave()
    file_processor.save_data(path=".//data_tmp//embedding_df.pkl", data=embedding_feats)

    # Step 3: TF-IDF features
    # ------------------------
    # MAX_FEATS = 150
    # corpus_tfidf_pos = [" ".join(item) for item in corpus_pos]
    # tfidf_pos = compute_tfidf_feats(corpus=corpus_tfidf_pos, max_feats=MAX_FEATS)
    # df_tfidf_pos = pd.DataFrame(tfidf_pos, columns=["tfidf_pos_{}".format(i) for i in range(MAX_FEATS)])

    # MAX_FEATS = 150
    # corpus_tfidf_acc = [" ".join(item) for item in corpus_acc]
    # tfidf_acc = compute_tfidf_feats(corpus=corpus_tfidf_acc, max_feats=MAX_FEATS)
    # df_tfidf_acc = pd.DataFrame(tfidf_acc, columns=["tfidf_acc_{}".format(i) for i in range(MAX_FEATS)])

    # tfidf_feats = pd.concat([df_tfidf_pos, df_tfidf_acc], axis=1)
    # tfidf_feats["fragment_id"] = fragment_id

    ##########################################################################
    total_feats = pd.merge(total_feats, embedding_feats, on="fragment_id", how="left")
    # total_feats = pd.merge(total_feats, tfidf_feats, on="fragment_id", how="left")

    train_feats = total_feats[total_feats["behavior_id"].notnull()]
    test_feats = total_feats[total_feats["behavior_id"].isnull()].drop("behavior_id", axis=1).reset_index(drop=True)

    n_folds = 5
    scores, importances, oof_pred, y_pred = lightgbm_classifier_training(train_df=train_feats, 
                                                                          test_df=test_feats,
                                                                          id_name="fragment_id",
                                                                          target_name="behavior_id",
                                                                          stratified=True, 
                                                                          shuffle=True,
                                                                          n_classes=19,
                                                                          n_folds=n_folds)
    # clf_pred_to_submission(y_valid=oof_pred, y_pred=y_pred, score=scores,
    #                         target_name="behavior_id", id_name="fragment_id",
    #                         sub_str_field="lgb_{}".format(n_folds), save_oof=False)
