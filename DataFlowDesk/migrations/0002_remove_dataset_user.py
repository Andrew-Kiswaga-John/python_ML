# Generated by Django 5.1.3 on 2024-11-20 13:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DataFlowDesk', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='dataset',
            name='user',
        ),
    ]