from classifier.lib.converter import Word2Index
import tensorflow as tf


class DatasetGenerator:
    """Datasetジェネレータ。querysetを受け取りas_datasetメソッドで全てのデータの埋め込みベクトルとカテゴリのタプルが入った
    tf.data.Datasetオブジェクトをリターンする。
    """

    def __init__(self, queryset, training: bool = True, remove_stopwords: bool = False):
        self._embeddings = list()
        self._category = list()
        self._training = training
        self._converter = Word2Index(remove_stopwords)
        self._prepare(queryset)
        self._converter.save()

    def _prepare(self, queryset):
        self._n_data = 0
        for news in queryset:
            self._n_data += 1
            self._embeddings.append(
                self._converter.encode(news.text, training=self._training))
            self._category.append(tf.constant(
                int(news.category), dtype=tf.int32))
        self._embeddings = [tf.constant(self._converter.decode(
            embedding), dtype=tf.int32) for embedding in self._embeddings]

    @property
    def model_input(self):
        return self._converter.n_vocab, max(self._category) + 1

    @ property
    def output_signature(self):
        signature = (tf.TensorSpec(shape=(self._converter.n_vocab,), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.int32))
        return signature

    def as_dataset(self):
        return tf.data.Dataset.from_tensor_slices((self._embeddings, self._category))
