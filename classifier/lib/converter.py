from typing import Any, List
from janome.tokenizer import Tokenizer
import numpy as np
import pickle
from abc import ABC, abstractmethod
import os
from functools import partial
from classifier import OpenType, TC, DataPath


def not_empty_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def pickle_save(object: Any, path: str):
    with open(path, mode=OpenType.WRITE) as file:
        pickle.dump(object, file)


def pickle_load(path):
    with open(path, mode=OpenType.READ) as file:
        obj = pickle.load(file)
    return obj


class Dictionary:
    """辞書オブジェクト。辞書に新しい単語を追加したり単語と単語indexとのマップ機能を持つ。
    """

    def __init__(self, remove_stopwords: bool = False):
        # string to index
        self._dictionary = dict()
        # index to string
        self._inverse_dictionary = dict()
        # stopwords flag
        if remove_stopwords:
            with open(DataPath.STOPWORDS, mode=OpenType.READTEXT) as file:
                self._stopwords = file.read().split(os.linesep)
        else:
            self._stopwords = None

    @property
    def n_vocab(self) -> int:
        return len(self._dictionary)

    def append(self, new_word: str):
        new_index = self.n_vocab
        assert new_word not in self._dictionary
        if self._stopwords is None or new_word not in self._stopwords:
            self._dictionary[new_word] = new_index
            self._inverse_dictionary[new_index] = new_word

    def word2index(self, word):
        if word in self._dictionary:
            return self._dictionary[word]
        else:
            return None

    def index2word(self, index):
        assert index in self._inverse_dictionary
        return self._inverse_dictionary[index]

    def has(self, word):
        return word in self._dictionary


class Converter(ABC):
    """前処理オブジェクト。テキストをモデルが読み取れるベクトルに変更する。複数の前処理が考えられるので
    抽象既定クラスを一応用意。

    """
    @abstractmethod
    def encode(self, text: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, embedding: Any) -> np.ndarray:
        raise NotImplementedError


class Word2Index(Converter):
    """テキストを辞書に存在する各単語を含む/含まないを示すバイナリのベクトルに変換するConverter。
    名詞のみを抽出する。
    セーブメソッドで内部の辞書を既定のファイルにセーブすることができ、
    インスタンス生成時は既定のファイルから自動でロードする。

    """

    def __init__(self, remove_stopwords: bool = False):
        self._tokenizer = Tokenizer()
        self._dictionary: Dictionary = None
        self._remove_stopwords = remove_stopwords
        self.load()

    def _nounlist(self, text: str):
        """Textを受け取ると名詞のみを抽出しリストにして返す

        Args:
            text (str): 被分類用のテキスト

        Returns:
            list: 名詞のリスト
        """
        nounlist = list()
        for token in self._tokenizer.tokenize(text):
            if token.extra is not None and \
                token.extra[TC.PART_OF_SPEECH_INDEX] \
                    .split(TC.SPLITTER)[TC.PRIMARY] == TC.NOUN:
                nounlist.append(str(token.extra[-3]))
        return nounlist

    def encode(self, text: str, training=True) -> List[int]:
        """テキストを受け取ると名詞のみを抽出し、辞書を利用して単語インデックスに変換したもののリストを返す。
        未知の単語を見つけた場合はトレーニング時には辞書に追加する。
        Args:
            text (str): 被分類用のテキスト
            training (bool, optional): トレーニングフラグ。

        Returns:
            List[int]: 単語インデックスのリスト
        """
        noun_list = self._nounlist(text)
        encode = partial(self._encode_word, training=training)
        return list(map(encode, noun_list))

    def _encode_word(self, word, training=True):
        if not self._dictionary.has(word) and training:
            self._dictionary.append(word)
        return self._dictionary.word2index(word)

    def decode(self, embedding: List[int]) -> np.ndarray:
        """encodeメソッドによって生成された単語インデックスのリストを受け取ると辞書内の全ての単語に対し
        テキストに含まれる(1)、含まれない(0)の情報を格納したベクトルを返す。

        Args:
            embedding (List[int]): 単語インデックスのリスト

        Returns:
            np.ndarray: 辞書内の全ての単語に対しテキストに含まれる(1)、含まれない(0)の情報を格納したベクトル
        """
        shape = (self._dictionary.n_vocab,)
        vector = np.zeros(shape, dtype=np.int)
        for index in embedding:
            if index is not None:
                vector[index] = 1
        return vector

    @property
    def n_vocab(self):
        return self._dictionary.n_vocab

    def save(self):
        if self._remove_stopwords:
            path = DataPath.DICT_S
        else:
            path = DataPath.DICT

        pickle_save(self._dictionary, path)

    def load(self):
        if self._remove_stopwords:
            path = DataPath.DICT_S
        else:
            path = DataPath.DICT

        if not_empty_file(path):
            self._dictionary = pickle_load(path)
        else:
            self._dictionary = Dictionary(self._remove_stopwords)
