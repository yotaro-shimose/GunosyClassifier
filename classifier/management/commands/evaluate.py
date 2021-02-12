import tensorflow as tf
from classifier.lib.generator import DatasetGenerator
from classifier.lib.naivebayes import NaiveBayesClassifier
from classifier.models import News
from django.core.management.base import BaseCommand
from classifier import Evaluation


class Command(BaseCommand):
    help = 'データセット全体から8割をトレーニングデータに、２割をテストデータにして精度をプリントする'

    def add_arguments(self, parser):
        parser.add_argument('-s', '--stopwords', nargs='?',
                            default=False, type=bool)
        parser.add_argument('-i', '--initcount',
                            nargs=1, default=1, type=int)

    def handle(self, stopwords: bool = False,
               initcount: int = 1, *args, **options):
        batch_size = Evaluation.BATCH_SIZE
        test_ratio = Evaluation.TEST_RATIO
        queryset = News.objects.all()
        data_size = len(queryset)
        generator = DatasetGenerator(queryset, remove_stopwords=stopwords)

        model = NaiveBayesClassifier(
            *generator.model_input, init_count=initcount)
        dataset: tf.data.Dataset = generator.as_dataset()
        dataset = dataset.shuffle(data_size)
        test_size = int(data_size * test_ratio)
        training_dataset = dataset.skip(test_size)
        test_dataset = dataset.take(test_size)

        for x, y in training_dataset.batch(batch_size):
            model.train(x, y)

        predictions = list()
        targets = list()
        for x, y in test_dataset.batch(batch_size):
            pred_y = model(x)
            predictions.append(pred_y)
            targets.append(y)
        predictions = tf.concat(predictions, axis=0)
        targets = tf.concat(targets, axis=0)
        accuracy = tf.reduce_sum(tf.cast(targets == predictions,
                                         dtype=tf.int32)) / targets.shape[0]
        confusion_matrix = tf.math.confusion_matrix(targets, predictions)
        print('accuracy: {}'.format(accuracy))
        print('confusion_matrix: {}'.format(confusion_matrix))
