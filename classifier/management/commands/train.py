from classifier.models import News
from classifier.lib.naivebayes import NaiveBayesClassifier
from django.core.management.base import BaseCommand
from classifier.lib.generator import DatasetGenerator
import tensorflow as tf
from classifier import DataPath, Evaluation


class Command(BaseCommand):
    help = 'naive bayesをトレーニングし既定のファイルにSavedModel形式でモデルを保存する'

    def add_arguments(self, parser):
        parser.add_argument('--stopwords', nargs='?', default=False, type=bool)

    def handle(self, stopwords: bool = False, *args, **options):
        batch_size = Evaluation.BATCH_SIZE
        if stopwords:
            path = DataPath.MODEL_S
        else:
            path = DataPath.MODEL
        queryset = News.objects.all()
        generator = DatasetGenerator(
            queryset, training=True, remove_stopwords=stopwords)
        model = NaiveBayesClassifier(*generator.model_input)

        dataset: tf.data.Dataset = generator.as_dataset()
        for x, y in dataset.batch(batch_size):
            model.train(x, y)
        model.save(path)
