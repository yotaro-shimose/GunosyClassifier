import requests
from bs4 import BeautifulSoup
import time
import logging

from classifier.models import News
from classifier import Scraiping, ErrorMessage


class IllegalStructureException(Exception):
    pass


def get_html(url):
    """urlから、httpリクエストを行い、HTML文を取得する

    Args:
        url (str): リクエスト先URL

    Returns:
        [BeautifulSoup]: 取得したHTML情報
    """
    time.sleep(Scraiping.INTERVAL_TIME)
    response = requests.get(url)
    return BeautifulSoup(response.text, Scraiping.PARSE_HTML_COMMAND)


def get_text(url):
    """ url先の記事のタイトル、本文を取得する

    Args:
        url (str): リクエスト先URL

    Raises:
        IllegalStructureException: HTML構成がグノシーのニュース記事と異なることを示す例外

    Returns:
        [str]: 記事のタイトルと本文を繋ぎ合わせた文章
    """
    # htmlの取得
    bs = get_html(url)

    # タイトル、本文を取得
    title = bs.select_one(Scraiping.NEWS_TITLE_CLASS)
    text_list = bs.select(Scraiping.NEWS_BODY_CLASS)

    # グノシーのURLかチェック
    if title is None or len(text_list) == Scraiping.EMPTY_LIST_LENGTH:
        raise IllegalStructureException(ErrorMessage.ILLEGAL_URL_MESSAGE)

    # データ整形
    text_list = [text.text for text in text_list]
    return title.text + Scraiping.EMPTY_STRING.join(text_list)


def create_news(url_list, text_list, category):
    """ ニュース情報をDBに一括登録する

    Args:
        url_list (list): 各記事のURLリスト
        text_list (list): 各記事のタイトルと本文を繋げたリスト
        category (str): 当該記事のカテゴリー
    """

    # Newsオブジェクトに変換
    news_list = [News(url=url, text=text, category=category)
                 for url, text in zip(url_list, text_list)]

    # Newsテーブルに一括登録
    News.objects.bulk_create(news_list)


def get_url_text_list(category_no_list):
    """対象カテゴリのURLと本文の一覧を取得する

    Args:
        category_no_list (list): アクセスする小カテゴリ一覧

    Returns:
        [tuple]: 各記事のURL一覧、タイトル＋本文一覧
    """
    # 初期化
    url_list = []
    text_list = []

    for category_no in category_no_list:
        for page in range(Scraiping.MAX_PAGE):
            # 記事一覧のHTML取得
            bs = get_html(Scraiping.BASE_URL.format(category_no, page+1))

            # HTMLから各記事を抽出
            title_list = bs.select(Scraiping.NEWS_LIST_CLASS)

            # 対象となる記事のURL一覧を生成
            temp_url_list = [text.get(Scraiping.HREF_COMMAND)
                             for text in title_list]
            url_list.extend(temp_url_list)

            # URL先から本文を取得
            try:
                text_list.extend([get_text(url)
                                  for url in temp_url_list])
            except IllegalStructureException as e:
                logging.warn(str(e))
                continue
    return url_list, text_list


def insert_news():
    """グノシーの記事一覧から、各記事をNewsテーブルに登録する

    """
    category_dict = Scraiping.CATEGORY_DICT
    for category_key in category_dict:
        # 今回のカテゴリ（エンタメ、スポーツ・・・）ごとの小カテゴリの一覧を取得
        category_no_list = category_dict[category_key]

        # 当該カテゴリにおけるすべての記事情報を取得
        url_list, text_list = get_url_text_list(category_no_list)

        # Newsテーブルに一括登録
        create_news(url_list, text_list, category_key)
