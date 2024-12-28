from django.contrib import admin
from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('<int:dataset_id>/dashboard/', views.dashboard, name='dataset_dashboard'),
    path('general_dashboard/', views.general_dashboard, name='general_dashboard'),
    path('', views.my_view, name='home'),
    path('', views.upload_page, name='upload_page'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('create-dataset/step1/', views.create_dataset_step1, name='create_dataset_step1'),
    path('create-dataset/step2/', views.create_dataset_step2, name='create_dataset_step2'),
    path('dataset/<int:id>/', views.display_dataset, name='display_dataset'),
    path('dataset/<int:dataset_id>/cleaning_preview/', views.data_cleaning_preview, name='data_cleaning_preview'),
    path('dataset/<int:dataset_id>/perform_cleaning/', views.perform_data_cleaning, name='perform_data_cleaning'),
    path('perform_data_normalization/<int:dataset_id>/', views.perform_data_normalization, name='perform_data_normalization'),
    path('dataset/<int:dataset_id>/graphs/', views.display_graphs, name='display_graphs'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('train_model/', views.train_model, name='train_model'),
    path('train_model/<int:dataset_id>/', views.train_model, name='train_model_with_dataset'),

    path('model_training/<int:id>/', views.training_page, name='model_training'),
    path('train_model_nn/', views.train_model_nn, name='train_model_nn'),
    path('generate_model_report/', views.generate_model_report, name='generate_model_report'),
    path('upload/', views.upload_file, name='upload'),
    path('my_datasets/', views.my_datasets, name='my_datasets'),
    path('save_graph/', views.save_graph, name='save_graph'),
    path('get_columns/', views.get_columns, name='get_columns'),
    path('get_columns_target/', views.get_columns_target, name='get_columns_target'),
    path('get_columns_graphs/', views.get_columns_graphs, name='get_columns_graphs'),
    # path('dataset/show_all/', views.all_datasets, name='all_datasets'),
    path('get_columns_graphs/', views.get_columns_graphs, name='get_columns_graphs'),
    path('dataset/<int:dataset_id>/graphs/', views.display_graphs, name='display_graphs'),
    path('save_graph/', views.save_graph, name='save_graph'),
    path('predictions/', views.render_predictions_view, name='predictions'),
    path('perform_predictions/', views.perform_predictions, name='perform_predictions'),
    path('fetch-columns/', views.fetch_columns, name='fetch_columns'),
    path('download_model/', views.download_model, name='download_model'),
    path('download_cleaned_data/<int:dataset_id>/', views.download_cleaned_data, name='download_cleaned_data'),

    # Predictions logic
    path('fetch-models/', views.fetch_models, name='fetch-models'),
    path('visualizations/', views.model_visualizations, name='model_visualizations'),
    path('fetch-visualizations/', views.fetch_visualizations, name='fetch_visualizations'),
    path('download-visualizations/', views.download_visualizations, name='download_visualizations'),
    path('fetch-model-details/', views.fetch_model_details, name='fetch_model_details'),
    path('make-prediction/', views.make_prediction, name='make_prediction'),
    path('model-predictions/', views.model_predictions, name='model_predictions'),

    path('tutorials/', views.tutorials, name='tutorials'),
    path('documentation/', views.documentation, name='documentation'),

    # Add these URL patterns to your urls.py
    path('auth/signin/', views.signin, name='login'),
    path('auth/signup/', views.signup, name='register'),
    path('auth/signout/', views.signout, name='signout'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)