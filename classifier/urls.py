from django.conf.urls import url
from . import views
from classifier import View
urlpatterns = [
    url(View.GUESS_URL, views.guess, name=View.GUESS_URL_NAME),
    url(View.HOME_URL, views.home, name=View.HOME_URL_NAME),
]
