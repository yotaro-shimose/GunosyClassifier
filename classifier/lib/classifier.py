import tensorflow as tf

from classifier import CATEGORY_MAP, DataPath
from classifier.lib.bfc import BertFeatureClassifier as BFC
from classifier.lib.converter import Word2Index

FIRST = 0


class Classifier:
    """推論時用の分類器。既定のフォルダから学習済みのモデルと前処理用のconverterをロードして保持する。
    classifyメソッドで分類を行う。
    """

    def __init__(self):
        path = DataPath.MODEL
        self._model = tf.keras.models.load_model(path)
        self._converter = Word2Index()

    def classify(self, text: str) -> str:
        """テキストを分類する。

        Args:
            text (str): 被分類用のテキスト

        Returns:
            str: 分類結果(like 'エンタメ')
        """
        embedding = tf.cast(self._converter.decode(
            self._converter.encode(text, training=False)), dtype=tf.int32)
        embedding = tf.expand_dims(embedding, axis=0)
        category = self._model(embedding)[FIRST]
        return CATEGORY_MAP[category]


class BertFeatureClassifier(Classifier):
    def __init__(self):
        self._bfc = BFC(load_clf=True)

    def classify(self, text: str) -> str:
        """テキストを分類する。

        Args:
            text (str): 被分類用のテキスト

        Returns:
            str: 分類結果(like 'エンタメ')
        """
        inputs = list()
        inputs.append(text)
        category = int(self._bfc.predict(inputs)[FIRST])
        return CATEGORY_MAP[category]
