from classifier import Evaluation
from classifier.lib.bfc import BertFeatureClassifier
from classifier.models import News
from django.core.management.base import BaseCommand
from django_pandas.io import read_frame
from sklearn.metrics import accuracy_score, confusion_matrix


class FieldName:
    TEXT = 'text'
    CATEGORY = 'category'


def load_dataset():
    queryset = News.objects.all()
    df = read_frame(queryset)
    # shuffle dataset
    df = df.sample(frac=1)
    texts = df[FieldName.TEXT]
    labels = df[FieldName.CATEGORY]
    test_ratio = Evaluation.TEST_RATIO
    test_size = int(len(df.index) * test_ratio)
    train_x, train_y = texts[test_size:], labels[test_size:]
    test_x, test_y = texts[:test_size], labels[:test_size]
    return train_x, train_y, test_x, test_y


class Command(BaseCommand):
    help = 'データセット全体から8割をトレーニングデータに、２割をテストデータにして精度をプリントする'

    def add_arguments(self, parser):
        parser.add_argument('-s', '--stopwords', nargs='?',
                            default=False, type=bool)
        parser.add_argument('-i', '--initcount',
                            nargs=1, default=1, type=int)

    def handle(self, *args, **options):
        model = BertFeatureClassifier()
        train_x, train_y, test_x, test_y = load_dataset()
        model.train(train_x, train_y)
        pred_y = model.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)
        cm = confusion_matrix(test_y, pred_y)
        print('accuracy: {}'.format(accuracy))
        print('confusion_matrix: {}'.format(cm))
