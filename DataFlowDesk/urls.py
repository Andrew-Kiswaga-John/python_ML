from django.contrib import admin
from django.urls import path
from . import views


from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.my_view, name='home'),
    path('upload/', views.upload_page, name='upload_page'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('create-dataset/step1/', views.create_dataset_step1, name='create_dataset_step1'),
    path('create-dataset/step2/', views.create_dataset_step2, name='create_dataset_step2'),
    # path('upload/', views.upload_file, name='upload_file'),
    path('dataset/<int:id>/', views.display_dataset, name='display_dataset'),

    # path('upload/', views.upload_file, name='upload_file'),

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)