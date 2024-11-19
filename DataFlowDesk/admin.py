from django.contrib import admin
from .models import Dataset, Profile, MLModel, ModelResult, ExportedFile, DataPreprocessingLog, Tutorial

# Register each model
admin.site.register(Dataset)
admin.site.register(Profile)
admin.site.register(MLModel)
admin.site.register(ModelResult)
admin.site.register(ExportedFile)
admin.site.register(DataPreprocessingLog)
admin.site.register(Tutorial)
