"""BFC module
BFC stands for Bert Feature Classifier
"""


import pickle
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertJapaneseTokenizer, TFBertModel
from transformers.tokenization_utils_base import BatchEncoding

from classifier import DataPath, Evaluation, OpenType


class Token:
    CLS = '[CLS]'
    SEP = '[SEP]'


class Kernel:
    RBF = 'rbf'


class TokenizerOptions:
    PADDING_STRATEGY = 'longest'
    TENSOR_TYPE = 'tf'


class ModelName:
    BERT_BASE_JAPANESE_WWM = 'cl-tohoku/bert-base-japanese-whole-word-masking'


class BertFeatureClassifier:
    """BERTを用いてテキストを784次元の特徴量に変換し
    この特徴量の上で古典的分類器をトレーニングするクラス
    """

    def __init__(self, load_clf=False):
        self._tokenizer = BertJapaneseTokenizer.from_pretrained(
            ModelName.BERT_BASE_JAPANESE_WWM)
        self._bert = TFBertModel.from_pretrained(
            ModelName.BERT_BASE_JAPANESE_WWM)
        if load_clf:
            self._clf = self._load_clf()
        else:
            self._clf = RandomForestClassifier()

    def _compute_embedding(self, texts: List[str]):
        encoded = list()
        n_iterations = len(texts) // Evaluation.BATCH_SIZE
        for i in range(n_iterations):
            minibatch = texts[i *
                              Evaluation.BATCH_SIZE: (i + 1) * Evaluation.BATCH_SIZE]
            minibatch_encoded: BatchEncoding = self._tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=minibatch,
                return_tensors=TokenizerOptions.TENSOR_TYPE,
                return_attention_mask=False,
                truncation=True,
                padding=TokenizerOptions.PADDING_STRATEGY,
            )
            encoded.append(minibatch_encoded.input_ids)
        token = tf.concat(encoded, axis=0)
        dataset = tf.data.Dataset.from_tensor_slices(
            token).batch(Evaluation.BATCH_SIZE)
        embedding = list()
        for token in dataset:
            embedding.append(self._bert(token).last_hidden_state)
        # B, N, 784
        embedding = tf.concat(embedding, axis=0)
        # B, 784
        embedding = tf.reduce_mean(embedding, axis=1)
        return embedding.numpy()

    def train(self, texts: List[str], labels: np.ndarray):
        embeddings = self._compute_embedding(texts)
        self._clf.fit(embeddings, labels)
        pred_train = self._clf.predict(embeddings)
        accuracy_train = accuracy_score(labels, pred_train)
        return accuracy_train

    def predict(self, texts: List[str]):
        embeddings = self._compute_embedding(texts)
        prediction = self._clf.predict(embeddings)
        return prediction

    def save(self):
        with open(DataPath.RANDOMFOREST, mode=OpenType.WRITE) as file:
            pickle.dump(self._clf, file)

    def _load_clf(self):
        with open(DataPath.RANDOMFOREST, mode=OpenType.READ) as file:
            clf = pickle.load(file)
        return clf
