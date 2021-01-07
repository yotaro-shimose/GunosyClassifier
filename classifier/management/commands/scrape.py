from django.core.management.base import BaseCommand
from classifier.lib.scraping import insert_news


class Command(BaseCommand):
    help = 'グノシーのニュース情報をスクレイピングし、Newsテーブルに登録する'

    def handle(self, *args, **options):
        insert_news()
