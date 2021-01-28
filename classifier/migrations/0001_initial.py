# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2020-12-29 16:43
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='News',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True,
                                        serialize=False, verbose_name='ID')),
                ('url', models.CharField(max_length=100)),
                ('text', models.TextField(null=True)),
                ('category', models.CharField(max_length=1, null=True)),
            ],
        ),
    ]
