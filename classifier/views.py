from classifier.lib.scraping import IllegalStructureException
from classifier.forms import GuessForm
from classifier import View, ErrorMessage
from django.shortcuts import render
from django.http import HttpResponse
from .domains import guess_category
import json
import logging


def home(request):
    """ホームのページにレンダリング

    Args:
        request (HttpRequest): Httpリクエスト情報

    Returns:
        [HttpResponse]: ホームのページ情報
    """
    form = GuessForm()
    context = {'form': form}
    return render(request, View.HOME_HTML_NAME, context)


def guess(request):
    """受け取ったリクエストからURL先の記事のカテゴリ名を予測するAPI

    Args:
        request (HttpRequest): Httpリクエスト情報

    Returns:
        HttpResponse: カテゴリ名の予測結果
    """
    form = GuessForm(request.GET)
    if form.is_valid():
        try:
            category = guess_category(request.GET.get(View.URL_KEY_NAME))
            params = {
                'category': category,
                'status': View.SUCCES_STATUS
            }
        except IllegalStructureException as e:
            logging.warn(e)
            params = {
                'message': {
                    View.URL_KEY_NAME: ErrorMessage.NO_GUNOSY_URL_MESSAGE
                },
                'status': View.FAILED_STATUS
            }
    else:
        logging.warn(form.errors)
        params = {
            'message': form.errors,
            'status': View.FAILED_STATUS
        }
    # json形式の文字列を生成
    json_str = json.dumps(params, ensure_ascii=False)
    return HttpResponse(json_str)
