from django.db import models
from classifier import Model


class News(models.Model):
    url = models.CharField(max_length=Model.URL_MAX_LENGTH)
    text = models.TextField(null=Model.TEXT_NULLABLE_FLAG)
    category = models.CharField(
        max_length=Model.CATEGORY_MAX_LENGTH,
        null=Model.CATEGORY_NULLABLE_FLAG
    )
