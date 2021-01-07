from classifier.lib.bfc import BertFeatureClassifier
from classifier.models import News
from django.core.management.base import BaseCommand
from django_pandas.io import read_frame


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
    return texts, labels


class Command(BaseCommand):
    help = 'データセット全体から8割をトレーニングデータに、２割をテストデータにして精度をプリントする'

    def add_arguments(self, parser):
        parser.add_argument('-s', '--stopwords', nargs='?',
                            default=False, type=bool)
        parser.add_argument('-i', '--initcount',
                            nargs=1, default=1, type=int)

    def handle(self, *args, **options):
        model = BertFeatureClassifier()
        texts, labels = load_dataset()
        model.train(texts, labels)
        model.save()
        print('Training Completed')
