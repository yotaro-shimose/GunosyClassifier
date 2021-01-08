import enum


class ClassifierIndex(enum.Enum):
    NAIVEBAYES = enum.auto()
    BFC = enum.auto()


class TC:
    """TokenizerConstantsの略

    """
    PART_OF_SPEECH_INDEX = 0
    PRIMARY = 0
    NOUN = "名詞"
    SPLITTER = ","


class DataPath:
    # 独自辞書のファイルパス
    DICT = './data/dictionary.dat'
    # ストップワード削除版独自辞書のファイルパス
    DICT_S = './data/dictionary_s.dat'
    # ナイーブベイズのTFSavedModelのファイルパス
    MODEL = './data/naivebayesmodel'
    # ストップワード削除版ナイーブベイズのTFSavedModelのファイルパス
    MODEL_S = './data/naivebayesmodel_s'
    # ストップワードリストのファイルパス
    STOPWORDS = './data/stopwords.txt'
    # Sklearnのランダムフォレストのファイルパス
    RANDOMFOREST = './data/randomforest.dat'


class OpenType:
    READ = 'rb'
    WRITE = 'wb'
    READTEXT = 'r'


class Scraiping:
    """スクレイピング処理で使用する定数クラス

    """
    # URL一覧を取得するベースのURL
    BASE_URL = 'https://gunosy.com/categories/{0}?page={1}'

    # 各カテゴリのURL番号辞書
    CATEGORY_DICT = {
        '0': [9, 10, 11, 12, 13, 14, 15, 17],
        '1': [18, 19, 20, 21, 22, 43, 44, 47, 48, 51, 54],
        '2': [23, 24, 25],
        '3': [26, 27, 28],
        '4': [29, 30],
        '5': [31, 32, 33, 42],
        '6': [34, 35, 36, 37, 52],
        '7': [38, 39, 40, 41]
    }

    # cssクラス名一覧
    # 記事タイトルクラス名
    NEWS_TITLE_CLASS = "[class='article_header_title']"

    # 記事本文クラス名
    NEWS_BODY_CLASS = "[class='article gtm-click'] p"

    # 記事URL一覧クラス名
    NEWS_LIST_CLASS = "[class='list_title'] a"

    # HTMLハイパーリンク構文
    HREF_COMMAND = 'href'

    # HTMLパースコマンド
    PARSE_HTML_COMMAND = 'html.parser'

    # スクレイピングするページの最大値
    MAX_PAGE = 5

    # サーバー負荷を考慮したリクエストインターバル時間
    # ToDo あとで1にする
    INTERVAL_TIME = 0.5

    # 空文字
    EMPTY_STRING = ''

    # 空のリスト長
    EMPTY_LIST_LENGTH = 0


class Form:
    """Form情報定数クラス

    """
    # URLテキストボックスのラベル名
    URL_LABEL_NAME = 'URL'

    # URLテキストボックスの最大長
    URL_MAX_LENGTH = 100

    # URL必須フラグ
    URL_REQUIRED_FLAG = True

    # URLプレースホルダー
    URL_PLACEHOLDER = 'https://xxxxx.com'

    # URLのCSSクラス名
    URL_CLASS_NAME = 'form-control col-sm-6'


class Model:
    """Model情報定数クラス

    """

    # URLカラムの最大長
    URL_MAX_LENGTH = 100

    # TEXTカラムNULL許可フラグ
    TEXT_NULLABLE_FLAG = True

    # CATEGORYカラムNULL許可フラグ
    CATEGORY_NULLABLE_FLAG = True

    # CATEGORYカラムの最大長
    CATEGORY_MAX_LENGTH = 1


class View:
    """View機能で使用する定数クラス

    """
    # リクエストが成功したことを示すステータス
    SUCCES_STATUS = 200

    # リクエストが失敗したことを示すステータス
    FAILED_STATUS = 400

    # ホームのテンプレートページ名
    HOME_HTML_NAME = 'home.html'

    # URLキー名
    URL_KEY_NAME = 'url'

    # 推測APIのURL
    GUESS_URL = 'guess'

    # 推測APIのURL名
    GUESS_URL_NAME = 'guess'

    # ホーム画面のURL
    HOME_URL = ''

    # ホーム画面のURL名
    HOME_URL_NAME = 'home'


class ErrorMessage:
    """エラーメッセージ定数クラス

    """
    # 必須エラーメッセージ
    REQUIRED_MESSAGE = 'URLを入力してください'

    # 404ページエラーメッセージ
    NO_EXIST_PAGE_MESSAGE = 'そのページは存在しません'

    # 不正URLエラーメッセージ
    ILLEGAL_URL_MESSAGE = '不正なHTML構成をしたURLです。'

    # グノシーのURLでないことを示すエラーメッセージ
    NO_GUNOSY_URL_MESSAGE = 'グノシーのニュース記事のURLを貼り付けてください'


class Evaluation:
    # Training時のBatchSize。メモリと速度のトレードオフ。最終的に出来上がるモデルには影響なし。
    BATCH_SIZE = 128
    # TestData/WholeDataの割合
    TEST_RATIO = 0.2


# カテゴリindexからカテゴリ文字列への逆引きリスト
CATEGORY_MAP = [
    'エンタメ',
    'スポーツ',
    'おもしろ',
    '国内',
    '海外',
    'コラム',
    'IT・科学',
    'グルメ',
]

CLASSIFIER_TYPE = ClassifierIndex.BFC
