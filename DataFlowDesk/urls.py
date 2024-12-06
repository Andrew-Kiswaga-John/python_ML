from django.contrib import admin
from django.urls import path
from . import views


from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.my_view, name='home'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('create-dataset/step1/', views.create_dataset_step1, name='create_dataset_step1'),
    path('create-dataset/step2/', views.create_dataset_step2, name='create_dataset_step2'),
    path('dataset/<int:id>/', views.display_dataset, name='display_dataset'),
    path('dataset/<int:dataset_id>/cleaning_preview/', views.data_cleaning_preview, name='data_cleaning_preview'),
    path('dataset/<int:dataset_id>/perform_cleaning/', views.perform_data_cleaning, name='perform_data_cleaning'),
    path('perform_data_normalization/<int:dataset_id>/', views.perform_data_normalization, name='perform_data_normalization'),
    path('dataset/<int:id>/graphs/', views.display_graphs, name='display_graphs'),

    path('train_model/', views.train_model, name='train_model'),
    path('model_training/', views.training_page, name='model_training'),

    path('get_columns/', views.get_columns, name='get_columns'),

    path('dataset/show_all/', views.all_datasets, name='all_datasets')

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)