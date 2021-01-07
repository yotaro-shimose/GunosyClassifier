from classifier import CLASSIFIER_TYPE, ClassifierIndex
from classifier.lib.scraping import get_text
from classifier.lib.classifier import Classifier, BertFeatureClassifier


if CLASSIFIER_TYPE == ClassifierIndex.NAIVEBAYES:
    classifier = Classifier()
elif CLASSIFIER_TYPE == ClassifierIndex.BFC:
    classifier = BertFeatureClassifier()


def guess_category(url):
    """url先の記事情報から、カテゴリを推測する

    Args:
        url (str): 記事情報URL

    Returns:
        (str): 予測結果のカテゴリ名
    """
    # 受け取ったurl先のHTMLからタイトル、本文を取得する
    text = get_text(url)

    # 取得したtextからurl先の記事のカテゴリを予測する
    category = classifier.classify(text)
    return category
