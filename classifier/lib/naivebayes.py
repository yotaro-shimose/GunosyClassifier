import tensorflow as tf


class VariableName:
    COUNTS = 'counts'
    CATEGORY_COUNTS = 'category_counts'


class Equation:
    CROSS = '...i, ...j -> ...ji'
    DOT = '...i, ji -> ...ji'


class ArgName:
    N_VOCAB = 'n_vocab'
    N_CATEGORY = 'n_category',
    INIT_COUNT = 'init_count'


@tf.keras.utils.register_keras_serializable()
class NaiveBayesClassifier(tf.keras.Model):
    """ナイーブベイズ分類器クラス。
    単語がTextに含まれるかどうかを示したバイナリの埋め込みベクトルを受け取ると、Textの分類番号をリターンする。
    埋め込みベクトルとラベルのテンソルをtrainメソッドに入力することでトレーニングを行う。saveメソッドでSavedModelを保存できる。

    """

    def __init__(
        self,
        n_vocab: int,
        n_category: int,
        init_count: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._n_category = n_category
        self._n_vocab = n_vocab
        self._init_count = init_count
        shape = (n_category, n_vocab)
        initializer = tf.keras.initializers.Ones()
        # C, V
        self._counts: tf.Variable = self.add_weight(
            name=VariableName.COUNTS,
            shape=shape,
            dtype=tf.int32,
            initializer=initializer
        )
        self._counts.assign(self._counts * self._init_count)
        # C
        shape = (n_category,)
        self._category_counts: tf.Variable = self.add_weight(
            name=VariableName.CATEGORY_COUNTS,
            shape=shape,
            dtype=tf.int32,
            initializer=initializer
        )
        self._category_counts.assign(
            n_category * self._category_counts * self._init_count)

    def train(self, x: tf.Tensor, y: tf.Tensor):
        """x, yを用いて学習（同じ学習データを入力しないように注意）

        Args:
            x (tf.Tensor): shape (B, V) のバイナリTensor
            y (tf.Tensor): shape (B,) のラベルTensor
        """
        if not self.built:
            self(x)
        y = tf.one_hot(y, depth=self._n_category, dtype=tf.int32)
        # 重み初期化時にラプラススムージング
        # C, V
        counts = tf.reduce_sum(tf.einsum(Equation.CROSS, x, y), axis=0)
        self._counts.assign_add(counts)
        # C
        category_counts = tf.reduce_sum(y, axis=0)
        self._category_counts.assign_add(category_counts)

    def call(self, embedding: tf.Tensor) -> tf.Tensor:
        """単語の埋め込み情報からクラスを予測

        Args:
            embedding (tf.Tensor): shape (B, V) のバイナリTensor

        Returns:
            tf.Tensor: shape (B,) のクラスindexを格納したTensor
        """
        # C, 1
        divisor = tf.expand_dims(self._category_counts, axis=-1)
        # C, V
        probabilities = tf.cast(self._counts / divisor, tf.float32)
        # 1, C
        category_distribution = tf.cast(tf.expand_dims(
            self._category_counts / tf.reduce_sum(self._category_counts),
            axis=0), tf.float32)
        # B, V
        embedding = tf.cast(embedding, tf.float32)
        # B, C, V
        posteriors = tf.einsum(Equation.DOT, embedding, probabilities) + \
            tf.einsum(Equation.DOT, 1. - embedding, 1. - probabilities)
        # B, C
        log_posterior = tf.math.reduce_sum(tf.math.log(
            posteriors), axis=-1) + tf.math.log(category_distribution)
        predicts = tf.math.argmax(
            log_posterior, axis=-1, output_type=tf.int32)
        return predicts

    def get_config(self):
        return super().get_config().update({
            ArgName.N_VOCAB: self._n_vocab,
            ArgName.N_CATEGORY: self._n_category,
            ArgName.INIT_COUNT: self._init_count
        })
