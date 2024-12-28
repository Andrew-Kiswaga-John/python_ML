from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from django.conf import settings
from datetime import datetime
from .models import Dataset, MLModel, DataPreprocessingLog, Profile, ModelResult
from django.db.models.functions import TruncMonth
from django.db.models import Avg
from django.db import models
import json
import matplotlib.pyplot as plt
import io
import time
import base64
from matplotlib.backends.backend_agg import FigureCanvas
from django.shortcuts import render
from django.shortcuts import render, redirect
from .forms import DatasetMetaForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import Profile

# Add scikit-learn imports
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import logging
import zipfile
from django.http import HttpResponse
from .models import Dataset, MLModel, DataPreprocessingLog, ModelResult
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.forms import modelform_factory
from .models import Dataset
import csv
import os
import uuid
import os
import csv
from django.shortcuts import render, redirect
from .models import Dataset
from django.shortcuts import render, redirect
from django.urls import reverse  # For generating URLs
import pandas as pd
from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
from django.shortcuts import render, redirect
import plotly.express as px

import pandas as pd
from .models import Dataset

import pandas as pd
from .models import Dataset

from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from django.utils.timezone import now
from django.utils.timezone import now
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from django.utils.timezone import now



import pandas as pd
from .models import Dataset

import pandas as pd
from .models import Dataset
from django.contrib import messages
from django.shortcuts import render, get_object_or_404
import os
import logging
import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from .models import Dataset


from io import StringIO
from django.core.files.base import ContentFile
import uuid
import os
import pylightxl
import pandas as pd
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from django.shortcuts import get_object_or_404
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import joblib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from typing import List, Tuple, Optional, Dict


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')




# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import logging

logger = logging.getLogger(__name__)







def my_datasets(request):
    datasets = Dataset.objects.all().order_by('-uploaded_at')
    
    datasets_list = []
    for dataset in datasets:
        # Get the file size in a readable format
        if dataset.file_path:
            size = dataset.file_path.size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
        else:
            size_str = "N/A"
            
        # Get the number of rows and columns
        try:
            file_path = dataset.cleaned_file.path if dataset.cleaned_file else dataset.file_path.path
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            rows = format(len(df), ',d')  # Format with commas for thousands
            columns = format(len(df.columns), ',d')
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            rows = "N/A"
            columns = "N/A"
            
        datasets_list.append({
            'id': dataset.id,
            'name': dataset.name,
            'description': dataset.description,
            'created_at': dataset.uploaded_at,
            'size': size_str,
            'rows': rows,
            'columns': columns,
            'status': dataset.status
        })
    
    context = {
        'datasets': datasets_list,
        'page_title': 'My Datasets'
    }
    return render(request, 'datasets/show_datasets.html', context)


@ensure_csrf_cookie
def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'success': True, 'redirect_url': '/general_dashboard/'})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid credentials'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})



def general_dashboard(request):
    if not request.user.is_authenticated:
        return redirect('signin')
        
    # Get all datasets for the current user
    datasets = Dataset.objects.filter(user=request.user)
    
    # Basic Statistics with safety checks
    total_datasets = datasets.count() or 0
    clean_datasets = datasets.filter(status='processed').count() or 0
    unclean_datasets = max(0, total_datasets - clean_datasets)
    total_models = MLModel.objects.count() or 0
    
    # Calculate percentages with safety checks
    clean_datasets_percentage = (clean_datasets / total_datasets * 100) if total_datasets > 0 else 0
    unclean_datasets_percentage = (unclean_datasets / total_datasets * 100) if total_datasets > 0 else 0
    
    # Calculate usage percentage for each dataset with safety check
    if datasets.exists():
        for dataset in datasets:
            usage_count = MLModel.objects.filter(dataset=dataset).count()
            max_usage = MLModel.objects.count() or 1  # Avoid division by zero
            dataset.usage_percentage = (usage_count / max_usage) * 100
    
    # Get current and previous month for comparison
    current_date = timezone.now()
    current_month_start = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    previous_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    
    # Dataset categories analysis (current month) with safety checks
    current_month_stats = {
        'Diagnostic Imaging': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='diagnostic imaging'
        ).count() or 0,
        'Data.Gov': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='data.gov'
        ).count() or 0,
        'Image Net': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='image net'
        ).count() or 0,
        'MNIST': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='mnist'
        ).count() or 0,
        'MHLDDS': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='mhldds'
        ).count() or 0,
        'HES': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='hes'
        ).count() or 0,
    }

    # Previous month statistics with safety checks
    previous_month_stats = {}
    for category, _ in current_month_stats.items():
        previous_month_stats[category] = datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains=category.lower()
        ).count() or 0
    
    # Quality and Completeness Analysis
    last_6_months = current_date - timedelta(days=180)
    monthly_stats = datasets.filter(
        uploaded_at__gte=last_6_months
    ).annotate(
        month=TruncMonth('uploaded_at')
    ).values('month').annotate(
        total=Count('id'),
        clean=Count('id', filter=models.Q(status='processed'))
    ).order_by('month')
    
    quality_labels = []
    completeness_data = []
    quality_data = []
    
    for stat in monthly_stats:
        quality_labels.append(stat['month'].strftime('%b %Y'))
        total = stat['total'] or 1  # Avoid division by zero
        clean = stat['clean']
        completeness = (clean / total * 100) if total > 0 else 0
        quality = completeness * 0.8  # Simplified quality metric
        completeness_data.append(completeness)
        quality_data.append(quality)
    
    # If no monthly stats, provide default empty month data
    if not quality_labels:
        for i in range(6):
            month_date = current_date - timedelta(days=30 * i)
            quality_labels.append(month_date.strftime('%b %Y'))
            completeness_data.append(0)
            quality_data.append(0)
    
    # Generate matplotlib visualizations with safety checks
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 10})
    
    # 1. Missing Values Bar Chart
    missing_data = datasets.filter(status='processed').count() or 0
    total_data = datasets.count() or 0
    
    fig_missing = plt.figure(figsize=(10, 6))
    plt.bar(['Complete Data', 'Missing Data'], 
           [total_data - missing_data, missing_data],
           color=['#10B981', '#EF4444'])
    plt.title('Data Completeness Overview', pad=20)
    plt.ylabel('Number of Datasets')
    for i, v in enumerate([total_data - missing_data, missing_data]):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    dataset_dist_plot = fig_to_base64(fig_missing)
    plt.close(fig_missing)
    
    # 2. Data Types Distribution Pie Chart
    fig_types = plt.figure(figsize=(10, 6))
    data_types = {
        'CSV Files': datasets.filter(file_path__endswith='.csv').count() or 0,
        'Excel Files': (datasets.filter(file_path__endswith='.xlsx').count() + 
                       datasets.filter(file_path__endswith='.xls').count()) or 0,
        'Text Files': datasets.filter(file_path__endswith='.txt').count() or 0,
    }
    
    # Add default value if no data
    if sum(data_types.values()) == 0:
        data_types = {'No Data': 1}
    
    plt.pie(data_types.values(), labels=data_types.keys(), autopct='%1.1f%%', 
        colors=['#4F46E5', '#10B981', '#F59E0B'],
        explode=[0.05] * len(data_types))
    plt.title('Dataset Types Distribution', pad=20)
    plt.axis('equal')
    quality_trend_plot = fig_to_base64(fig_types)
    plt.close(fig_types)

    # Dataset Usage Statistics with safety checks
    datasets_usage = []
    if datasets.exists():
        for dataset in datasets:
            usage_count = MLModel.objects.filter(dataset=dataset).count()
            if usage_count > 0:
                datasets_usage.append({
                    'title': dataset.name,
                    'usage_percentage': min((usage_count / (total_datasets or 1) * 100), 100),
                    'count': usage_count
                })
    
    # Sort datasets_usage by count in descending order
    datasets_usage = sorted(datasets_usage, key=lambda x: x['count'], reverse=True)[:10]
    
    # 3. Dataset Usage Heatmap
    if datasets_usage:
        usage_data = [[d['usage_percentage']] for d in datasets_usage[:8]]
        labels = [d['title'] for d in datasets_usage[:8]]
        
        fig_heatmap = plt.figure(figsize=(12, 4))
        plt.imshow([usage_data[0]], cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Usage %')
        plt.yticks(range(len(labels)), labels)
        plt.xticks([])
        plt.title('Dataset Usage Intensity', pad=20)
        plt.tight_layout()
        usage_heatmap = fig_to_base64(fig_heatmap)
        plt.close(fig_heatmap)
    else:
        usage_heatmap = None
    
    # 4. Processing Status Donut Chart
# 4. Processing Status Donut Chart
    fig_status = plt.figure(figsize=(8, 8))
    if total_datasets > 0:
        plt.pie([clean_datasets, unclean_datasets], 
            labels=['Clean', 'Unclean'], 
            autopct='%1.1f%%',
            colors=['#10B981', '#EF4444'],
            wedgeprops=dict(width=0.5, edgecolor='white'),
            textprops={'color': '#374151', 'fontsize': 12})
    else:
        # Fix: Changed autopct format to be a proper string format
        plt.pie([1], 
            labels=['No Data'], 
            autopct='%1.1f%%',  # This is the fixed format
            colors=['#CBD5E0'],
            wedgeprops=dict(width=0.5, edgecolor='white'),
            textprops={'color': '#374151', 'fontsize': 12})
    plt.title('Dataset Processing Status', pad=20)
    processing_status_plot = fig_to_base64(fig_status)
    plt.close(fig_status)
    
    # Calculate average completeness and quality with safety checks
    avg_completeness = sum(completeness_data) / len(completeness_data) if completeness_data else 0
    avg_quality = sum(quality_data) / len(quality_data) if quality_data else 0
    
    # Model Performance Metrics with safety checks
    model_metrics = MLModel.objects.all()
    model_performance = [0, 0, 0, 0, 0]  # Default values
    
    if model_metrics.exists():
        total_models = 0
        metrics_sum = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0}
        
        for model in model_metrics:
            if model.results:
                try:
                    results = model.results
                    if isinstance(results, str):
                        results = json.loads(results)
                    
                    metrics_sum['accuracy'] += float(results.get('accuracy', 0))
                    metrics_sum['precision'] += float(results.get('precision', 0))
                    metrics_sum['recall'] += float(results.get('recall', 0))
                    metrics_sum['f1_score'] += float(results.get('f1_score', 0))
                    metrics_sum['roc_auc'] += float(results.get('roc_auc', 0))
                    total_models += 1
                except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                    continue
        
        if total_models > 0:
            model_performance = [
                (metrics_sum['accuracy'] / total_models) * 100,
                (metrics_sum['precision'] / total_models) * 100,
                (metrics_sum['recall'] / total_models) * 100,
                (metrics_sum['f1_score'] / total_models) * 100,
                (metrics_sum['roc_auc'] / total_models) * 100
            ]
    
    # Recent Activities with safety checks
    recent_activities = []
    
    # Dataset activities
    dataset_activities = datasets.order_by('-uploaded_at')[:5]
    for dataset in dataset_activities:
        recent_activities.append({
            'timestamp': dataset.uploaded_at,
            'description': f'Dataset "{dataset.name}" was uploaded'
        })
    
    # Model training activities
    model_activities = MLModel.objects.order_by('-created_at')[:5]
    for model in model_activities:
        recent_activities.append({
            'timestamp': model.created_at,
            'description': f'Model was trained on dataset "{model.dataset.name}"'
        })
    
    # Sort activities by timestamp
    recent_activities = sorted(recent_activities, key=lambda x: x['timestamp'], reverse=True)[:5]
    
    # Get trained models data with safety checks
    trained_models = MLModel.objects.filter(dataset__user=request.user).select_related('dataset').order_by('-created_at')[:3]
    models_data = []
    
    for model in trained_models:
        model_results = ModelResult.objects.filter(model=model).order_by('-generated_at')
        
        metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None
        }
        
        for result in model_results:
            metrics[result.metric_name.lower()] = result.metric_value

        model_data = {
            'id': model.id,
            'name': model.algorithm,
            'dataset_name': model.dataset.name,
            'dataset_id': model.dataset.id,
            'type': 'classification' if 'class' in model.algorithm.lower() else 'regression',
            'created_at': model.created_at,
            **metrics
        }
        models_data.append(model_data)

    # Group models by dataset with safety checks
    dataset_results = {}
    for model_data in models_data:
        dataset_id = model_data['dataset_id']
        if dataset_id not in dataset_results:
            dataset_results[dataset_id] = {
                'dataset_name': model_data['dataset_name'],
                'models': []
            }
        dataset_results[dataset_id]['models'].append(model_data)
    available_dataset = Dataset.objects.filter(user=request.user)

    context = {
        'datasets': datasets,
        'total_datasets': total_datasets,
        'clean_datasets': clean_datasets,
        'unclean_datasets': unclean_datasets,
        'total_models': total_models,
        'clean_datasets_percentage': round(clean_datasets_percentage, 1),
        'unclean_datasets_percentage': round(unclean_datasets_percentage, 1),
        'completeness': round(avg_completeness, 1),
        'quality': round(avg_quality, 1),
        'datasets_analysis_current': list(current_month_stats.values()),
        'datasets_analysis_previous': list(previous_month_stats.values()),
        'quality_labels': quality_labels,
        'completeness_data': completeness_data,
        'quality_data': quality_data,
        'datasets_usage': datasets_usage,
        'dataset_dist_plot': dataset_dist_plot,
        'quality_trend_plot': quality_trend_plot,
        'usage_heatmap': usage_heatmap,
        'processing_status_plot': processing_status_plot,
        'model_performance': model_performance,
        'recent_activities': recent_activities,
        'trained_models': models_data,
        'dataset_results': dataset_results,
        'available_dataset' : available_dataset
    }
    
    return render(request, "general_dashboard.html", context)



def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@ensure_csrf_cookie
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if User.objects.filter(username=username).exists():
            return JsonResponse({'success': False, 'error': 'Username already exists'})
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({'success': False, 'error': 'Email already exists'})
        
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            Profile.objects.create(user=user)
            
            # Automatically log in the user after signup
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
            
            return JsonResponse({'success': True, 'redirect_url': '/general_dashboard/'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def download_cleaned_data(request, dataset_id):
    # Get the dataset
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check if cleaned file exists
    if not dataset.cleaned_file:
        return HttpResponse("No cleaned data available for download", status=404)
    
    # Open the file and create the response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{dataset.name}_cleaned.csv"'
    
    # Read the cleaned file and write to response
    with open(dataset.cleaned_file.path, 'r', encoding='utf-8') as file:
        response.write(file.read())
    
    return response

def dashboard(request,dataset_id):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)
    
    try:
        if dataset.status == 'processed':
            # Load cleaned file (CSV only)
            data = pd.read_csv(dataset.cleaned_file)
        else:
            # Handle CSV and Excel files for unprocessed datasets
            file_path = dataset.file_path.path
            if file_path.endswith('.csv'):
                try:
                    data = pd.read_csv(file_path, on_bad_lines='skip')  # Default UTF-8
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(file_path, on_bad_lines='skip', encoding='ISO-8859-1')  # Fallback to Latin-1
                    except Exception as e:
                        return JsonResponse({'error': f'CSV File reading error: {str(e)}'})
            elif file_path.endswith('.xlsx'):
                try:
                    data = pd.read_excel(file_path, engine='openpyxl')  # Use openpyxl for .xlsx
                except Exception as e:
                    return JsonResponse({'error': f'Excel (.xlsx) File reading error: {str(e)}'})
            elif file_path.endswith('.xls'):
                try:
                    data = pd.read_excel(file_path, engine='xlrd')  # Use xlrd for .xls
                except Exception as e:
                    return JsonResponse({'error': f'Excel (.xls) File reading error: {str(e)}'})
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'})
    except Exception as e:
        return JsonResponse({'error': f'Error loading dataset: {str(e)}'})

    dataset_preview = data.head(10).values.tolist()  # Show the first 10 rows
    columns = data.columns.tolist()

    try:
        # Calculate metrics
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        missing_percentage = (total_missing / (data.shape[0] * data.shape[1])) * 100
        
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        categorical_columns = data.attrs.get('categorical_columns', [])
        
        feature_types = {
            'Numerical': len(data.select_dtypes(include=['int64', 'float64']).columns) - len(categorical_columns),
            'Categorical': len(categorical_columns),
            'DateTime': len(data.select_dtypes(include=['datetime64']).columns),
            'Boolean': len(data.select_dtypes(include=['bool']).columns)
        }
        
        memory_usage = data.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage / (1024 * 1024)

        # Set the style for all plots
        plt.style.use('default')
        
        # 1. Target Distribution Plot
        fig_target = plt.figure(figsize=(8, 6))
        target_column = dataset.target_class
        if target_column and target_column in data.columns:
            target_counts = data[target_column].value_counts()
            sns.barplot(x=target_counts.index, y=target_counts.values)
            plt.title(f'Distribution of {target_column}', pad=20)
            plt.xticks(rotation=45)
            plt.xlabel(target_column)
            plt.ylabel('Count')
            plt.tight_layout()
            target_dist_plot = fig_to_base64(fig_target)
            plt.close(fig_target)
        else:
            target_dist_plot = None

        # 2. Feature Types Distribution
        fig_types = plt.figure(figsize=(8, 6))
        if sum(feature_types.values()) > 0:
            plt.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%', 
                   colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
            plt.title('Distribution of Feature Types', pad=20)
            plt.axis('equal')
            feature_types_plot = fig_to_base64(fig_types)
        else:
            feature_types_plot = None
        plt.close(fig_types)

        # 3. Data Quality Summary (Missing Values and Duplicates)
        fig_quality = plt.figure(figsize=(12, 5))
        
        # First subplot for data completeness
        plt.subplot(1, 2, 1)
        plt.pie([100 - missing_percentage, missing_percentage], 
                labels=['Complete', 'Missing'], 
                autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'])
        plt.title('Data Completeness')

        # Second subplot for data uniqueness
        plt.subplot(1, 2, 2)
        plt.pie([100 - duplicate_percentage, duplicate_percentage], 
                labels=['Unique', 'Duplicate'], 
                autopct='%1.1f%%', 
                colors=['#3498db', '#e67e22'])
        plt.title('Data Uniqueness')
        
        plt.tight_layout()
        quality_plot = fig_to_base64(fig_quality)
        plt.close(fig_quality)

        # 4. Correlation Overview (Top 10 Most Correlated Features)
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 1:
            correlation_matrix = data[numerical_cols].corr()
            
            # Calculate average absolute correlation for each feature
            avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
            
            # Take top 10 features or all if less than 10
            n_features = min(10, len(avg_correlations))
            
            fig_corr = plt.figure(figsize=(10, 6))
            sns.barplot(x=avg_correlations.values[:n_features], 
                       y=avg_correlations.index[:n_features],
                       palette='viridis')
            plt.title('Top Most Correlated Features\n(Average Absolute Correlation)', pad=20)
            plt.xlabel('Average Absolute Correlation')
            plt.tight_layout()
            correlation_plot = fig_to_base64(fig_corr)
            plt.close(fig_corr)
        else:
            correlation_plot = None

        return render(request, 'dashboard.html', {
            'dataset': dataset,
            'dataset_preview': dataset_preview,
            'columns': columns,
            'rows': data,
            'total_missing': total_missing,
            'missing_percentage': round(missing_percentage, 2),
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_percentage, 2),
            'feature_types': feature_types,
            'memory_usage_mb': round(memory_usage_mb, 2),
            'target_dist_plot': target_dist_plot,
            'feature_types_plot': feature_types_plot,
            'quality_plot': quality_plot,
            'correlation_plot': correlation_plot,
        })

    except Exception as e:
        return JsonResponse({'error': f'Error loading dataset: {str(e)}'})

def signout(request):
    logout(request)
    return redirect('home')

def create_dataset_step1(request):
    if request.method == "POST":
        form = DatasetMetaForm(request.POST)
        if form.is_valid():
            # Gather general dataset details
            dataset_meta = form.cleaned_data
            num_columns = dataset_meta['num_columns']

            # Gather column-specific details
            columns = []
            for i in range(num_columns):
                col_name = request.POST.get(f'col_name_{i}')
                col_type = request.POST.get(f'col_type_{i}')
                columns.append({'name': col_name, 'type': col_type})

            # Save metadata in session for Step 2
            dataset_meta['columns'] = columns
            request.session['dataset_meta'] = dataset_meta
            return redirect('create_dataset_step2')  # Redirect to Step 2
    else:
        form = DatasetMetaForm()
    return render(request, 'home.html', {'form': form})




def create_dataset_step2(request):
    if request.method == "POST":
        try:
            # Get all the form data
            data = request.POST.getlist('data')
            dataset_meta = json.loads(request.POST.get('dataset_meta', '{}'))
            target_class = request.POST.get('target_class')  # Get target class from form
            if not dataset_meta:
                return JsonResponse({"error": "Missing dataset metadata"}, status=400)

            columns = dataset_meta['columns']
            num_rows = int(dataset_meta['num_rows'])
            num_cols = len(columns)

            # Validate that we have the correct number of data points
            if len(data) != num_rows * num_cols:
                return JsonResponse({
                    "error": f"Expected {num_rows * num_cols} data points, but received {len(data)}"
                }, status=400)

            # Initialize validated data structure
            validated_data = []
            error_messages = []

            # Process data row by row
            for row in range(num_rows):
                row_data = []
                for col in range(num_cols):
                    value = data[row * num_cols + col]
                    column = columns[col]
                    
                    try:
                        # Validate based on column type
                        if column['type'] == 'int':
                            validated_value = int(value)
                        elif column['type'] == 'float':
                            validated_value = float(value)
                        elif column['type'] == 'string':
                            validated_value = str(value)
                        elif column['type'] == 'date':
                            validated_value = datetime.strptime(value, "%Y-%m-%d").date()
                        elif column['type'] == 'time':
                            validated_value = datetime.strptime(value, "%H:%M:%S").time()
                        else:
                            validated_value = str(value)
                            
                        row_data.append(validated_value)
                    except ValueError:
                        error_messages.append(
                            f"Row {row + 1}, Column '{column['name']}': Invalid {column['type']} value '{value}'"
                        )

                if not error_messages:  # Only add the row if there were no errors
                    validated_data.append(row_data)

            if error_messages:
                return JsonResponse({"error": "\n".join(error_messages)}, status=400)

            # Ensure the datasets directory exists
            datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
            os.makedirs(datasets_dir, exist_ok=True)

            # Create the CSV file
            file_path = os.path.join(datasets_dir, f"{dataset_meta['name']}.csv")
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ['ID'] + [col['name'] for col in columns]
                writer.writerow(header)
                
                # Write data rows
                for row_index, row_data in enumerate(validated_data, 1):
                    writer.writerow([row_index] + row_data)

            # Save to database
            new_dataset = Dataset.objects.create(
                user=request.user,
                name=dataset_meta['name'],
                description=dataset_meta.get('description', ''),
                file_path=file_path,  # Path relative to MEDIA_ROOT
                columns_info=columns,
                target_class=target_class,
                status="uploaded"
            )

            return JsonResponse({
                "message": "Dataset created successfully",
                "dataset_id": new_dataset.id
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    # If not POST, redirect to step 1
    return redirect('create_dataset_step1')


def display_dataset(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    # Load dataset file
    try:
        if dataset.status == 'processed':
            # Load cleaned file (CSV only)
            data = pd.read_csv(dataset.cleaned_file)
        else:
            # Handle CSV and Excel files for unprocessed datasets
            file_path = dataset.file_path.path
            if file_path.endswith('.csv'):
                try:
                    data = pd.read_csv(file_path, on_bad_lines='skip')  # Default UTF-8
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(file_path, on_bad_lines='skip', encoding='ISO-8859-1')  # Fallback to Latin-1
                    except Exception as e:
                        return JsonResponse({'error': f'CSV File reading error: {str(e)}'})
            elif file_path.endswith('.xlsx'):
                try:
                    data = pd.read_excel(file_path, engine='openpyxl')  # Use openpyxl for .xlsx
                except Exception as e:
                    return JsonResponse({'error': f'Excel (.xlsx) File reading error: {str(e)}'})
            elif file_path.endswith('.xls'):
                try:
                    data = pd.read_excel(file_path, engine='xlrd')  # Use xlrd for .xls
                except Exception as e:
                    return JsonResponse({'error': f'Excel (.xls) File reading error: {str(e)}'})
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'})
    except Exception as e:
        return JsonResponse({'error': f'Error loading dataset: {str(e)}'})

    # Prepare data for display
    dataset_preview = data.head(10).values.tolist()  # Show the first 10 rows
    columns = data.columns.tolist()

    # Render the template
    return render(request, 'datasets/display_dataset.html', {
        'dataset': dataset,
        'dataset_data': dataset_preview,
        'columns': columns,
        'stats': data.describe().to_dict(),  # Optional summary statistics
    })


def perform_data_normalization(request, dataset_id):
    # Ensure only POST requests are allowed for normalization
    if request.method == 'POST':
        try:
            # Fetch the dataset
            dataset = Dataset.objects.get(id=dataset_id)

            # Check if dataset is processed
            if dataset.status != 'processed':
                return JsonResponse({'error': 'Dataset not processed. Please clean the dataset first.'}, status=400)
            
            # Access the actual file path using .path
            file_path = dataset.cleaned_file.path

            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Identify numeric columns for normalization
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

            # Perform Standard Scaling
            scaler = StandardScaler()
            df[numeric_columns] = pd.DataFrame(
                scaler.fit_transform(df[numeric_columns]),
                columns=numeric_columns
            )

            # Save to a new file with a timestamp to avoid overwriting
            # new_file_path = f"{file_path.replace('.csv', '')}_normalized_{now().strftime('%Y%m%d%H%M%S')}.csv"
            # df.to_csv(new_file_path, index=False)

            cleaned_datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'normalized')
            os.makedirs(cleaned_datasets_dir, exist_ok=True)
            cleaned_file_name = os.path.basename(file_path).replace('_cleaned.csv', '_normalized.csv')
            cleaned_file_path = os.path.join(cleaned_datasets_dir, cleaned_file_name)
            df.to_csv(cleaned_file_path, index=False)

            # Update the dataset object to reflect the cleaned dataset
            dataset.cleaned_file = os.path.relpath(cleaned_file_path, settings.MEDIA_ROOT)

            dataset.save()

            # Log the preprocessing action in DataPreprocessingLog
            DataPreprocessingLog.objects.create(
                dataset=dataset,
                action='Data Normalized',
                parameters={
                    'scaler': 'StandardScaler',
                    'columns': list(numeric_columns)
                },
                timestamp=now()
            )

            # Return success response
            return JsonResponse({'message': f'Data normalized successfully. Saved as {cleaned_file_path}'})

        except Dataset.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)



@csrf_exempt
def get_columns_graphs(request):
    logging.debug(f"get_columns_graphs called with request.GET: {request.GET}")  # Add debug logging
    
    try:
        # Check both GET and POST for dataset_id
        dataset_id = request.GET.get('dataset_id') or request.POST.get('dataset_id')
        logging.debug(f"Received dataset_id: {dataset_id}")  # Debug log
        
        if not dataset_id:
            logging.warning("No dataset ID provided")  # Warning log
            return JsonResponse({
                'success': False,
                'error': 'No dataset ID provided'
            })
        
        # Convert dataset_id to integer
        try:
            dataset_id = int(dataset_id)
        except ValueError:
            logging.error(f"Invalid dataset ID format: {dataset_id}")  # Error log
            return JsonResponse({
                'success': False,
                'error': 'Invalid dataset ID format'
            })
            
        try:
            dataset = Dataset.objects.get(pk=dataset_id)
            logging.debug(f"Found dataset: {dataset.name}")  # Debug log
        except Dataset.DoesNotExist:
            logging.error(f"Dataset not found with ID: {dataset_id}")  # Error log
            return JsonResponse({
                'success': False,
                'error': f'Dataset with ID {dataset_id} not found'
            })
            
        # Read from cleaned file if available, otherwise from original file
        try:
            if dataset.cleaned_file:
                file_path = dataset.cleaned_file.path
                logging.debug(f"Using cleaned file: {file_path}")  # Debug log
            elif dataset.file_path:
                file_path = dataset.file_path.path
                logging.debug(f"Using original file: {file_path}")  # Debug log
            else:
                logging.error("No file found for dataset")  # Error log
                return JsonResponse({
                    'success': False,
                    'error': 'No file found for this dataset'
                })
                
            try:
                # Try different encodings
                encodings = ['utf-8', 'ISO-8859-1', 'latin1']
                df = None
                last_error = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                        logging.debug(f"Successfully read file with encoding: {encoding}")  # Debug log
                        break
                    except UnicodeDecodeError:
                        last_error = f"Failed to read with encoding: {encoding}"
                        continue
                    except Exception as e:
                        last_error = str(e)
                        break
                
                if df is None:
                    raise Exception(f"Failed to read file with any encoding. Last error: {last_error}")
                
            except Exception as e:
                logging.error(f"Error reading file: {str(e)}")  # Error log
                return JsonResponse({
                    'success': False,
                    'error': f'Error reading file: {str(e)}'
                })
            
            # Filter out text-heavy columns and get column types
            columns = []
            column_types = {}
            
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        # Check if it's a text-heavy column
                        if df[col].str.len().mean() <= 50:
                            columns.append(col)
                            column_types[col] = 'categorical'
                    else:
                        columns.append(col)
                        column_types[col] = 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'other'
                except Exception as e:
                    logging.warning(f"Error processing column {col}: {str(e)}")
                    continue
            
            logging.debug(f"Found columns: {columns}")  # Debug log
            
            return JsonResponse({
                'success': True,
                'columns': columns,
                'column_types': column_types
            })
            
        except Exception as e:
            logging.error(f"Error accessing file: {str(e)}")  # Error log
            return JsonResponse({
                'success': False,
                'error': f'Error accessing file: {str(e)}'
            })
            
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")  # Error log
        return JsonResponse({
            'success': False,
            'error': str(e)
        })




@csrf_exempt
def display_graphs(request, dataset_id=None):
    logging.debug(f"display_graphs called with dataset_id: {dataset_id}")
    
    if not dataset_id:
        return render(request, 'display_graphs.html', {
            'error': 'No dataset ID provided'
        })

    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        logging.debug(f"Found dataset: {dataset.name}")

        if request.method == 'POST':
            logging.debug('Processing POST request for graph generation')
            try:
                # Validate POST data
                x_column = request.POST.get('x_column')
                y_column = request.POST.get('y_column')
                graph_type = request.POST.get('graph_type', 'bar_chart')
                
                logging.debug(f'Parameters - X: {x_column}, Y: {y_column}, Type: {graph_type}')
                
                if not x_column:
                    return JsonResponse({
                        'success': False,
                        'error': 'X-axis column must be provided'
                    })

                if graph_type != 'histogram' and not y_column:
                    return JsonResponse({
                        'success': False,
                        'error': 'Y-axis column must be provided for non-histogram graphs'
                    })

                # Get file path
                if dataset.cleaned_file:
                    file_path = dataset.cleaned_file.path
                elif dataset.file_path:
                    file_path = dataset.file_path.path
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'No file found for this dataset'
                    })

                # Read dataset with multiple encoding attempts
                encodings = ['utf-8', 'ISO-8859-1', 'latin1']
                df = None
                last_error = None

                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                        logging.debug(f"Successfully read file with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        last_error = f"Failed to read with encoding: {encoding}"
                        continue
                    except Exception as e:
                        last_error = str(e)
                        break

                if df is None:
                    raise Exception(f"Failed to read file with any encoding. Last error: {last_error}")

                try:
                    # Determine column types
                    x_is_cat = df[x_column].dtype == 'object' or len(df[x_column].unique()) <= 10
                    y_is_count = y_column == 'count'
                    y_is_cat = False if y_is_count else (df[y_column].dtype == 'object' or len(df[y_column].unique()) <= 10)

                    logging.debug(f'Column types - X categorical: {x_is_cat}, Y count: {y_is_count}, Y categorical: {y_is_cat}')

                    # Set style with a valid style name
                    plt.style.use('default')
                    
                    # Create figure with a white background
                    plt.figure(figsize=(12, 8), facecolor='white')
                    plt.clf()

                    # Set the color palette
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    if y_is_count:
                        # Handle count visualization
                        value_counts = df[x_column].value_counts()
                        if graph_type == 'pie_chart' and len(value_counts) <= 10:
                            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                                  colors=colors[:len(value_counts)])
                            plt.title(f'Distribution of {x_column}', pad=20, size=14)
                            plt.axis('equal')
                        else:
                            ax = value_counts.plot(kind='bar', color=colors[0])
                            plt.title(f'Count of {x_column}', pad=20, size=14)
                            plt.xlabel(x_column, size=12)
                            plt.ylabel('Count', size=12)
                            
                            # Add value labels on top of bars
                            for i, v in enumerate(value_counts):
                                ax.text(i, v, str(v), ha='center', va='bottom')
                    else:
                        if graph_type == 'box_plot' and not x_is_cat and not y_is_cat:
                            sns.boxplot(x=x_column, y=y_column, data=df, color=colors[0])
                            plt.title(f'Box Plot of {y_column} by {x_column}', pad=20, size=14)
                        
                        elif graph_type == 'line_chart' and not x_is_cat and not y_is_cat:
                            plt.plot(df[x_column], df[y_column], marker='o', color=colors[0])
                            plt.title(f'Line Chart of {y_column} vs {x_column}', pad=20, size=14)
                            plt.xticks(rotation=45)
                        
                        elif graph_type == 'scatter_chart' and not x_is_cat and not y_is_cat:
                            plt.scatter(df[x_column], df[y_column], color=colors[0], alpha=0.5)
                            plt.title(f'Scatter Plot of {y_column} vs {x_column}', pad=20, size=14)
                        
                        elif graph_type == 'histogram':
                            plt.hist(df[x_column], bins=30, color=colors[0], alpha=0.7)
                            plt.title(f'Histogram of {x_column}', pad=20, size=14)
                            plt.xlabel(x_column)
                            plt.ylabel('Frequency')
                        
                        elif graph_type == 'bar_chart':
                            if x_is_cat:
                                ax = sns.barplot(x=x_column, y=y_column, data=df, color=colors[0])
                                plt.title(f'Bar Chart of {y_column} by {x_column}', pad=20, size=14)
                                
                                # Add value labels on top of bars
                                for i in ax.containers:
                                    ax.bar_label(i, padding=3)
                            else:
                                ax = df[y_column].value_counts().plot(kind='bar', color=colors[0])
                                plt.title(f'Bar Chart of {y_column}', pad=20, size=14)
                                
                                # Add value labels
                                for i, v in enumerate(df[y_column].value_counts()):
                                    ax.text(i, v, str(v), ha='center', va='bottom')
                        
                        elif graph_type == 'pie_chart' and x_is_cat and len(df[x_column].unique()) <= 10:
                            grouped_data = df.groupby(x_column)[y_column].sum()
                            plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%',
                                  colors=colors[:len(grouped_data)])
                            plt.title(f'Pie Chart of {y_column} by {x_column}', pad=20, size=14)
                            plt.axis('equal')
                        
                        else:
                            logging.debug('Invalid selection, defaulting to bar chart')
                            ax = sns.barplot(x=x_column, y=y_column, data=df, color=colors[0])
                            plt.title(f'Bar Chart of {y_column} by {x_column}', pad=20, size=14)
                            
                            # Add value labels
                            for i in ax.containers:
                                ax.bar_label(i, padding=3)

                    # Common plot settings
                    plt.xlabel(x_column, size=12)
                    plt.ylabel(y_column if not y_is_count else 'Count', size=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()

                    # Save plot to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    plt.close('all')  # Close all figures to free memory

                    # Convert to base64
                    graphic = base64.b64encode(image_png).decode('utf-8')

                    return JsonResponse({
                        'success': True,
                        'graphic': graphic
                    })

                except Exception as e:
                    logging.error(f'Error creating plot: {str(e)}')
                    return JsonResponse({
                        'success': False,
                        'error': f'Error creating plot: {str(e)}'
                    })

            except Exception as e:
                logging.error(f'Error processing POST request: {str(e)}')
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })

        # Handle GET request
        try:
            # Get file path
            if dataset.cleaned_file:
                file_path = dataset.cleaned_file.path
            elif dataset.file_path:
                file_path = dataset.file_path.path
            else:
                return render(request, 'display_graphs.html', {
                    'error': 'No file found for this dataset',
                    'dataset': dataset,
                    'dataset_id': dataset_id
                })

            # Read dataset
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

            # Get plottable columns
            plottable_columns = []
            column_types = {}

            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        if df[col].str.len().mean() <= 50:
                            plottable_columns.append(col)
                            column_types[col] = 'categorical'
                    else:
                        plottable_columns.append(col)
                        column_types[col] = 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'other'
                except Exception as e:
                    logging.warning(f"Error processing column {col}: {str(e)}")
                    continue

            # Sort saved graphs
            saved_graphs = sorted(dataset.graphs, key=lambda x: x.get('created_at', ''), reverse=True) if dataset.graphs else []

            context = {
                'dataset': dataset,
                'dataset_id': dataset_id,
                'columns': plottable_columns,
                'column_types': column_types,
                'saved_graphs': saved_graphs,
                'graph_types': [
                    ('bar_chart', 'Bar Chart'),
                    ('line_chart', 'Line Chart'),
                    ('scatter_chart', 'Scatter Plot'),
                    ('pie_chart', 'Pie Chart'),
                    ('histogram', 'Histogram'),
                    ('box_plot', 'Box Plot')
                ]
            }

            return render(request, 'display_graphs.html', context)

        except Exception as e:
            logging.error(f"Error preparing display context: {str(e)}")
            return render(request, 'display_graphs.html', {
                'error': str(e),
                'dataset': dataset,
                'dataset_id': dataset_id
            })

    except Dataset.DoesNotExist:
        logging.error(f"Dataset not found with ID: {dataset_id}")
        return render(request, 'display_graphs.html', {
            'error': f'Dataset with ID {dataset_id} not found'
        })
    except Exception as e:
        logging.error(f"Unexpected error in display_graphs: {str(e)}")
        return render(request, 'display_graphs.html', {
            'error': str(e)
        })
    

@csrf_exempt
def save_graph(request):
    logging.debug("save_graph view called")
    
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Only POST method is allowed'
        })

    try:
        # Get and validate required parameters
        graph_data = request.POST.get('graph_image')
        dataset_id = request.POST.get('dataset_id')
        x_column = request.POST.get('x_column')
        y_column = request.POST.get('y_column')
        graph_type = request.POST.get('graph_type')

        logging.debug(f"Received parameters - Dataset ID: {dataset_id}, X: {x_column}, Y: {y_column}, Type: {graph_type}")

        # Validate required fields
        if not all([graph_data, dataset_id, x_column, graph_type]):
            missing_fields = []
            if not graph_data: missing_fields.append('graph_image')
            if not dataset_id: missing_fields.append('dataset_id')
            if not x_column: missing_fields.append('x_column')
            if not graph_type: missing_fields.append('graph_type')
            
            return JsonResponse({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            })

        # Validate dataset_id format
        try:
            dataset_id = int(dataset_id)
        except ValueError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid dataset ID format'
            })

        # Get dataset
        try:
            dataset = Dataset.objects.get(pk=dataset_id)
            logging.debug(f"Found dataset: {dataset.name}")
        except Dataset.DoesNotExist:
            logging.error(f"Dataset not found with ID: {dataset_id}")
            return JsonResponse({
                'success': False,
                'error': 'Dataset not found'
            })

        # Generate timestamp and clean filename components
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Clean filename components to remove special characters
        def clean_filename(s):
            return "".join(c for c in s if c.isalnum() or c in ('-', '_')).lower()

        x_column_clean = clean_filename(x_column)
        y_column_clean = clean_filename(y_column) if y_column else 'count'
        graph_type_clean = clean_filename(graph_type)

        # Create filename and paths
        filename = f"graph_{dataset_id}_{graph_type_clean}_{x_column_clean}_vs_{y_column_clean}_{timestamp}.png"
        relative_path = f'datasets/graphs/{filename}'
        
        # Ensure media directories exist
        graphs_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        file_path = os.path.join(graphs_dir, filename)
        logging.debug(f"Saving graph to: {file_path}")

        # Process and save the image
        try:
            # Remove data URL prefix if present
            if 'base64,' in graph_data:
                graph_data = graph_data.split('base64,')[1]

            # Decode and save the image
            try:
                image_data = base64.b64decode(graph_data)
            except Exception as e:
                logging.error(f"Error decoding base64 data: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid image data'
                })

            # Save the file
            with open(file_path, 'wb') as f:
                f.write(image_data)

            # Verify file was saved
            if not os.path.exists(file_path):
                raise Exception("File was not saved successfully")

            # Create graph info dictionary
            graph_info = {
                'file_path': relative_path,
                'graph_type': graph_type,
                'x_column': x_column,
                'y_column': y_column,
                'created_at': timestamp,
                'file_size': os.path.getsize(file_path)
            }

            # Update dataset's graphs list
            try:
                graphs = dataset.graphs or []
                graphs.append(graph_info)
                
                # Keep only the last 50 graphs to prevent excessive storage
                if len(graphs) > 50:
                    # Delete old graph files
                    for old_graph in graphs[:-50]:
                        old_file_path = os.path.join(settings.MEDIA_ROOT, old_graph['file_path'])
                        try:
                            if os.path.exists(old_file_path):
                                os.remove(old_file_path)
                        except Exception as e:
                            logging.warning(f"Error deleting old graph file: {str(e)}")
                    
                    # Keep only the latest 50 graphs
                    graphs = graphs[-50:]

                dataset.graphs = graphs
                dataset.save()
                logging.debug("Dataset updated successfully")

                return JsonResponse({
                    'success': True,
                    'message': 'Graph saved successfully',
                    'data': {
                        'filename': filename,
                        'file_path': relative_path,
                        'created_at': timestamp
                    }
                })

            except Exception as e:
                # If there's an error saving to the database, delete the saved file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as delete_error:
                    logging.error(f"Error deleting file after database error: {str(delete_error)}")
                
                raise Exception(f"Error updating dataset: {str(e)}")

        except Exception as e:
            logging.error(f"Error saving graph: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Error saving graph: {str(e)}'
            })

    except Exception as e:
        logging.error(f"Unexpected error in save_graph: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })



# from autoclean import autoclean



def my_view(request):
    return render(request, 'home.html')

@csrf_exempt
def upload_file(request):
    print("Upload file called")  # Debug log
    if request.method == 'POST':
        try:
            print("Request POST data:", request.POST)  # Debug log
            print("Request FILES:", request.FILES)  # Debug log
            
            name = request.POST.get('name')
            description = request.POST.get('description')
            target_class = request.POST.get('target_class')
            dataset_type = request.POST.get('dataset_type')
            source = request.POST.get('source')

            print(f"Name: {name}, Description: {description}, Target: {target_class}, Source: {source}")  # Debug log

            if not all([name, target_class]):
                missing = []
                if not name: missing.append('name')
                if not target_class: missing.append('target class')
                error_msg = f"Missing required fields: {', '.join(missing)}"
                print(f"Error: {error_msg}")  # Debug log
                return JsonResponse({'error': error_msg}, status=400)

            # Handling local file uploads
            if source == 'local':
                file = request.FILES.get('file')
                if not file:
                    return JsonResponse({'error': 'No file uploaded.'}, status=400)

                # Read the file
                try:
                    if file.name.endswith('.csv'):
                        try:
                            # Add low_memory=False to handle mixed types warning
                            df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
                        except UnicodeDecodeError:
                            try:
                                df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1', low_memory=False)
                            except Exception as e:
                                return JsonResponse({'error': f'CSV File reading error: {str(e)}'}, status=400)

                        # Handle cases where the delimiter isn't a comma
                        if df.empty or len(df.columns) == 1:
                            try:
                                df = pd.read_csv(file, delimiter=';', low_memory=False)
                            except Exception as e:
                                return JsonResponse({'error': f'CSV Parsing Error with fallback delimiter: {str(e)}'}, status=400)
                    elif file.name.endswith('.xlsx'):
                        df = pd.read_excel(file, engine='openpyxl')
                    elif file.name.endswith('.xls'):
                        df = pd.read_excel(file, engine='xlrd')
                    else:
                        return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'}, status=400)
                except Exception as e:
                    return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

            # Handling Kaggle dataset URL
            elif source == 'kaggle':
                kaggle_link = request.POST.get('kaggle_link')
                if not kaggle_link:
                    return JsonResponse({'error': 'No Kaggle link provided.'}, status=400)

                try:
                    import requests
                    import zipfile
                    from urllib.parse import urlparse
                    
                    # Create datasets directory
                    datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    # Download to a temporary zip file
                    zip_path = os.path.join(datasets_dir, 'temp_dataset.zip')
                    
                    # Download the file with proper headers
                    headers = {
                        'User-Agent': 'Mozilla/5.0',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                    }
                    
                    response = requests.get(kaggle_link, headers=headers, stream=True)
                    response.raise_for_status()
                    
                    # Save the downloaded zip file
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # List all files in the zip
                        files = zip_ref.namelist()
                        print(f"Files in zip: {files}")  # Debug log
                        
                        # Find the first CSV or Excel file
                        data_file = None
                        for file in files:
                            if file.endswith(('.csv', '.xlsx', '.xls')):
                                data_file = file
                                break
                        
                        if not data_file:
                            raise Exception("No CSV or Excel files found in the downloaded dataset")
                        
                        # Extract the file
                        zip_ref.extract(data_file, datasets_dir)
                        file_path = os.path.join(datasets_dir, data_file)

                    # Load the dataset with low_memory=False
                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False)
                        elif file_path.endswith(('.xls', '.xlsx')):
                            df = pd.read_excel(file_path)
                    except Exception as e:
                        return JsonResponse({'error': f'Error reading Kaggle dataset: {str(e)}'}, status=400)

                    # Create a File object from the downloaded file
                    with open(file_path, 'rb') as f:
                        file = ContentFile(f.read())
                        file.name = os.path.basename(file_path)

                    # Clean up temporary files
                    os.remove(zip_path)
                    os.remove(file_path)

                except Exception as e:
                    return JsonResponse({'error': f'Error processing Kaggle dataset: {str(e)}'}, status=400)
            else:
                return JsonResponse({'error': 'Invalid data source selected.'}, status=400)

            # Clean column names and print them for debugging
            df.columns = df.columns.str.strip()
            print("Available columns:", list(df.columns))
            print("Target class:", target_class)

            # Case-insensitive column matching
            column_map = {col.lower(): col for col in df.columns}
            target_class_lower = target_class.lower()
            
            if target_class_lower in column_map:
                # Use the original column name
                target_class = column_map[target_class_lower]
            else:
                return JsonResponse({
                    'error': f'Selected target class "{target_class}" is not a valid column. Available columns: {", ".join(df.columns)}'
                }, status=400)

            # Save the dataset
            dataset = Dataset.objects.create(
                user=request.user,
                name=name,
                description=description,
                file_path=file,
                target_class=target_class,
                dataset_type=dataset_type,
                columns_info={
                    'columns': list(df.columns),
                    'dtypes': {col: str(df[col].dtype) for col in df.columns}
                }
            )

            # Get preview data
            preview_html = df.head().to_html(classes='table table-bordered')

            return JsonResponse({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'dataset_id': dataset.id,
                'preview': preview_html,
                'redirect_url': f'/dashboard/{dataset.id}'
            })

        except Exception as e:
            print(f"Error in upload_file: {str(e)}")  # Debug log
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'upload.html')


def clean_dataset(df, delete_header=False):
    """
    Cleans the dataset:
    - Handles duplicates, missing values, outliers, encoding, normalization
    - Includes preprocessing for boolean columns
    """
    print("Starting dataset cleaning...")
    
    # Store original column information
    original_columns = df.columns.tolist()
    
    # Keep track of categorical columns and target column
    categorical_columns = []
    target_column = None

    # If user wants to delete header
    if delete_header:
        print("Deleting header and using first data row as new header...")
        
        # Check if target_class exists and store its index if it does
        target_column_index = None
        new_target_column = None
        if 'target_class' in df.columns:
            target_column_index = df.columns.get_loc('target_class')
            new_target_column = df.iloc[0, target_column_index]
            print(f"Found target_class column at index {target_column_index}")
        
        new_headers = df.iloc[0].values.tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = new_headers
        
        # Store the column mapping information
        df.attrs['original_columns'] = original_columns
        if target_column_index is not None:
            df.attrs['new_target_column'] = new_target_column
            df.attrs['target_column_index'] = target_column_index
            print(f"Header replaced with first data row. Target column renamed from 'target_class' to '{new_target_column}'")
        else:
            print("Header replaced with first data row. No target_class column found.")

    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - df.shape[0]
    print(f"Removed {removed_duplicates} duplicate rows.")

    # First identify if we have a class or target_class column
    target_column = None
    if 'class' in df.columns:
        target_column = 'class'
    elif 'target_class' in df.columns:
        target_column = 'target_class'

    # Process each column
    for col in df.columns:
        print(f"Processing column: {col} (Type: {df[col].dtype})")

        # Skip processing if this is the target column
        if col == target_column:
            print(f"Skipping encoding for target column: {col}")
            continue

        # Handle boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in boolean column '{col}' with 'False'.")
                df[col].fillna(False, inplace=True)

            print(f"Converting boolean column '{col}' to numeric (1 for True, 0 for False).")
            df[col] = df[col].astype(int)

        # Handle numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in numerical column '{col}' with median.")
                df[col].fillna(df[col].median(), inplace=True)


            # Handle outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Handling {outliers} outliers in column '{col}'. Replacing with median.")
                df[col] = np.where(
                    (df[col] < lower_bound) | (df[col] > upper_bound),
                    df[col].median(),
                    df[col]
                )

        # Handle categorical columns
        elif pd.api.types.is_object_dtype(df[col]):

            categorical_columns.append(col)  # Track this as a categorical column
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in categorical column '{col}' with 'Unknown'.")
                df[col].fillna('Unknown', inplace=True)

            # Ensure all entries are strings and strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Encode categorical data (skip for target column)
            
            if col != target_column:
                print(f"Encoding categorical column '{col}' using Label Encoding.")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        # Handle other types
        else:
            print(f"Skipping column '{col}' as it does not fit numeric, boolean, or categorical types.")

    # Store categorical column information in the DataFrame attributes
    df.attrs['categorical_columns'] = categorical_columns
    if target_column:
        df.attrs['target_column'] = target_column
    
    print("Dataset cleaning completed.")
    return df

def perform_data_cleaning(request, dataset_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

    try:
        # Parse the JSON body
        data = json.loads(request.body)
        delete_header = data.get('remove_first_row', False)  # Get user's choice about header deletion

        # Fetch the dataset
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file_path.path

        # Determine file type based on the extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path, on_bad_lines='skip')
            cleaned_file_suffix = '_cleaned.csv'
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
            cleaned_file_suffix = '_xlscleaned.csv'
        else:
            return JsonResponse({'error': 'Unsupported file format. Only CSV and Excel files are allowed.'}, status=400)

        # Perform cleaning with the delete_header parameter
        cleaned_df = clean_dataset(df, delete_header=delete_header)

        # Save the cleaned dataset with a new name
        cleaned_datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'cleaned')
        os.makedirs(cleaned_datasets_dir, exist_ok=True)
        cleaned_file_name = os.path.basename(file_path).replace('.csv', cleaned_file_suffix).replace('.xls', cleaned_file_suffix).replace('.xlsx', cleaned_file_suffix)
        cleaned_file_path = os.path.join(cleaned_datasets_dir, cleaned_file_name)
        
        # Before saving, capture the important attributes
        target_column = cleaned_df.attrs.get('target_column')
        categorical_columns = cleaned_df.attrs.get('categorical_columns', [])
        
        # Save the cleaned DataFrame
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Update the dataset object with the cleaned file path
        relative_cleaned_path = os.path.relpath(cleaned_file_path, settings.MEDIA_ROOT)
        dataset.cleaned_file = relative_cleaned_path
        dataset.file_path = relative_cleaned_path  # Update the main file path to point to cleaned file
        dataset.status = 'processed'
        
        # Update column information
        column_types = {col: str(cleaned_df[col].dtype) for col in cleaned_df.columns}
        dataset.column_info = json.dumps(column_types)
        
        # Update target class based on the captured information
        if target_column:
            dataset.target_class = target_column
        elif 'class' in cleaned_df.columns:
            dataset.target_class = 'class'
        elif 'target_class' in cleaned_df.columns:
            dataset.target_class = 'target_class'
        
        # Store categorical columns information
        if categorical_columns:
            dataset.categorical_columns = json.dumps(categorical_columns)
        
        dataset.save()

        # Log the changes
        print(f"Updated dataset {dataset_id}:")
        print(f"- Target class: {dataset.target_class}")
        print(f"- Column info: {dataset.column_info}")
        print(f"- File path: {relative_cleaned_path}")

        return JsonResponse({
            'message': 'Data cleaned and saved successfully!',
            'rows_affected': len(df) - len(cleaned_df),
            'new_columns': list(cleaned_df.columns),
            'new_target_class': dataset.target_class,
            'new_file_path': relative_cleaned_path  # Use string path instead of FieldFile
        })

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}")
        return JsonResponse({'error': f"Error during cleaning: {str(e)}"}, status=500)

def data_cleaning_preview(request, dataset_id):
    """
    Generates a preview of tasks needed for cleaning the dataset.
    """
    try:
        # Fetch dataset from the database
        dataset = Dataset.objects.get(id=dataset_id)

        # Check if dataset is already processed
        if dataset.status == 'processed':
            return JsonResponse({'tasks': ["Dataset already processed"]})

        file_path = dataset.file_path.path  # Path to the dataset

        # Determine file type based on the extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path, on_bad_lines='skip')
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format. Only CSV and Excel files are allowed.'}, status=400)

        # Identify cleaning tasks
        tasks = []
        tasks.append("You will have the option to delete the header and use the first data row as the new header.")
        if df.isnull().values.any():
            tasks.append(f"Found {df.isnull().sum().sum()} missing values. These will be handled.")
        if df.duplicated().sum() > 0:
            tasks.append(f"Found {df.duplicated().sum()} duplicate rows. These will be removed.")
        tasks.append("Columns with incorrect data types will be converted.")
        tasks.append("Outliers in numerical columns will be treated.")

        return JsonResponse({'tasks': tasks})
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f"Error processing file: {str(e)}"}, status=500)



from django.shortcuts import render, get_object_or_404


import pandas as pd
from django.shortcuts import render, get_object_or_404
from .models import Dataset
import pandas as pd
from django.shortcuts import render, get_object_or_404
from .models import Dataset  # Adjust based on your actual model import

from django.shortcuts import render, get_object_or_404
import pandas as pd
from django.shortcuts import render, get_object_or_404
from .models import Dataset

from django.shortcuts import render, get_object_or_404
import pandas as pd
import numpy as np
from collections import Counter



# Function to render the upload.html page
def upload_page(request):
    return render(request, 'upload.html')

@csrf_exempt
@csrf_exempt
def get_columns_target(request):
    print("get_columns_target called")  # Debug log
    if request.method == 'POST':
        try:
            if request.FILES.get('file'):
                # Handle file upload (existing code remains the same)
                file = request.FILES['file']
                print(f"Processing file: {file.name}")
                
                try:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file, encoding='utf-8')
                    elif file.name.endswith('.xlsx'):
                        df = pd.read_excel(file, engine='openpyxl')
                    elif file.name.endswith('.xls'):
                        df = pd.read_excel(file, engine='xlrd')
                    else:
                        return JsonResponse({
                            'error': 'Unsupported file format. Please upload a CSV (.csv) or Excel (.xls, .xlsx) file'
                        }, status=400)
                except Exception as e:
                    return JsonResponse({
                        'error': f'Error reading file: {str(e)}. Please ensure the file is not corrupted and in the correct format.'
                    }, status=400)
            else:
                # Handle Kaggle URL
                try:
                    data = json.loads(request.body)
                    kaggle_url = data.get('kaggle_url')
                    if not kaggle_url:
                        return JsonResponse({'error': 'Kaggle URL is required'}, status=400)
                    
                    print(f"Processing Kaggle URL: {kaggle_url}")  # Debug log
                    
                    # Create datasets directory if it doesn't exist
                    datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
                    os.makedirs(datasets_dir, exist_ok=True)
                    
                    # Download file directly using requests
                    try:
                        import requests
                        import zipfile
                        from urllib.parse import urlparse
                        
                        # Download to a temporary zip file
                        zip_path = os.path.join(datasets_dir, 'temp_dataset.zip')
                        
                        # Download the file with proper headers
                        headers = {
                            'User-Agent': 'Mozilla/5.0',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                        }
                        
                        response = requests.get(kaggle_url, headers=headers, stream=True)
                        response.raise_for_status()
                        
                        # Save the downloaded zip file
                        with open(zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Extract the zip file
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # List all files in the zip
                            files = zip_ref.namelist()
                            print(f"Files in zip: {files}")  # Debug log
                            
                            # Find the first CSV or Excel file
                            data_file = None
                            for file in files:
                                if file.endswith(('.csv', '.xlsx', '.xls')):
                                    data_file = file
                                    break
                            
                            if not data_file:
                                raise Exception("No CSV or Excel files found in the downloaded dataset")
                            
                            # Extract the file
                            zip_ref.extract(data_file, datasets_dir)
                            file_path = os.path.join(datasets_dir, data_file)
                        
                        # Try to read the file with different encodings
                        if file_path.endswith('.csv'):
                            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                            df = None
                            last_error = None
                            for encoding in encodings:
                                try:
                                    df = pd.read_csv(file_path, encoding=encoding)
                                    print(f"Successfully read CSV with encoding: {encoding}")  # Debug log
                                    break
                                except Exception as e:
                                    last_error = e
                                    continue
                            
                            if df is None:
                                raise Exception(f"Unable to read the CSV file with any supported encoding. Last error: {str(last_error)}")
                        elif file_path.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file_path)
                        
                        # Clean up temporary files
                        os.remove(zip_path)
                        os.remove(file_path)
                        
                    except requests.exceptions.RequestException as e:
                        return JsonResponse({'error': f'Error downloading dataset: {str(e)}'}, status=400)
                    except Exception as e:
                        return JsonResponse({'error': f'Error processing dataset: {str(e)}'}, status=400)
                    
                except json.JSONDecodeError:
                    return JsonResponse({'error': 'Invalid JSON data'}, status=400)
                except Exception as e:
                    print(f"Error processing Kaggle URL: {str(e)}")  # Debug log
                    return JsonResponse({'error': str(e)}, status=400)
            
            # Clean up column names
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
            
            # Get column names and their data types
            columns = list(df.columns)
            dtypes = {col: str(df[col].dtype) for col in columns}
            
            print(f"Found columns: {columns}")  # Debug log
            print(f"First few rows of data:")  # Debug log
            print(df.head())  # Debug log
            
            response_data = {
                'columns': columns,
                'dtypes': dtypes
            }
            print(f"Sending response: {response_data}")  # Debug log
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")  # Debug log
            return JsonResponse({'error': str(e)}, status=400)
    
    print("Invalid request received")  # Debug log
    return JsonResponse({'error': 'Invalid request'}, status=400)


from django.template.loader import render_to_string
from django.conf import settings
import tempfile
import os
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
import pandas as pd
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from django.template.loader import render_to_string
from django.conf import settings
import tempfile
import os
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
from .models import MLModel, ModelResult
import pandas as pd

@csrf_exempt
def generate_model_report(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')

        # Get the model and related information
        model = get_object_or_404(MLModel, id=model_id)
        dataset = model.dataset
        model_results = ModelResult.objects.filter(model=model)

        # Get feature count and sample count
        df = pd.read_csv(dataset.cleaned_file.path)
        feature_count = len(df.columns) - 1  # Excluding target column
        sample_count = len(df)

        # Prepare metrics dictionary
        metrics = {}
        for result in model_results:
            metrics[result.metric_name] = round(float(result.metric_value), 4)

        # Create PDF using reportlab
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="model_report_{model_id}.pdf"'

        # Create the PDF object using reportlab
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.utils import ImageReader
        
        doc = SimpleDocTemplate(response, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Model Training Report", title_style))
        story.append(Spacer(1, 12))

        # Timestamp
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Dataset Information
        story.append(Paragraph("Dataset Information", styles['Heading2']))
        dataset_data = [
            ["Name:", dataset.name],
            ["Description:", dataset.description],
            ["Target Class:", dataset.target_class],
            ["Dataset Type:", 'Classification' if 'class' in model.algorithm.lower() else 'Regression'],
            ["Feature Count:", str(feature_count)],
            ["Sample Count:", str(sample_count)]
        ]
        
        t = Table(dataset_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # Model Information
        story.append(Paragraph("Model Information", styles['Heading2']))
        model_data = [
            ["Algorithm:", model.algorithm],
            ["Training Status:", model.training_status],
            ["Created At:", model.created_at.strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        t = Table(model_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # Metrics
        if metrics:
            story.append(Paragraph("Performance Metrics", styles['Heading2']))
            metrics_data = [[key, str(value)] for key, value in metrics.items()]
            t = Table(metrics_data, colWidths=[2*inch, 4*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

        # Visualizations
        # First try to find visualizations in model-specific directory
        model_identifier = f'{model.algorithm.lower()}_{dataset.id}'
        viz_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations', model_identifier)
        
        # If not found, try to find any directory containing this pattern
        if not os.path.exists(viz_dir):
            base_viz_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations')
            if os.path.exists(base_viz_dir):
                for dir_name in os.listdir(base_viz_dir):
                    if model_identifier in dir_name:
                        viz_dir = os.path.join(base_viz_dir, dir_name)
                        break

        if os.path.exists(viz_dir):
            story.append(Paragraph("Model Visualizations", styles['Heading2']))
            for f in os.listdir(viz_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(viz_dir, f)
                    try:
                        # Get a nice title from the filename
                        title = f.replace('_', ' ').replace('.png', '').replace('.jpg', '').replace('.jpeg', '').title()
                        
                        # Add title for the visualization
                        story.append(Paragraph(title, styles['Heading3']))
                        story.append(Spacer(1, 12))
                        
                        # Add the image with error handling
                        try:
                            # First try with PIL/Pillow
                            img = Image(img_path)
                            # Scale the image while maintaining aspect ratio
                            aspect = img.imageWidth / float(img.imageHeight)
                            if aspect > 1:
                                img.drawWidth = 6*inch
                                img.drawHeight = (6*inch) / aspect
                            else:
                                img.drawHeight = 4*inch
                                img.drawWidth = 4*inch * aspect
                        except:
                            # Fallback to ImageReader
                            img = Image(ImageReader(img_path))
                            img.drawWidth = 6*inch
                            img.drawHeight = 4*inch
                        
                        story.append(img)
                        story.append(Spacer(1, 20))
                    except Exception as e:
                        error_msg = f"Error loading image {f}: {str(e)}"
                        print(error_msg)  # Log the error
                        story.append(Paragraph(error_msg, styles['Normal']))
                        story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)
        return response

    except Exception as e:
        print(f"Error generating report: {str(e)}")  # Log the error
        return JsonResponse({'error': str(e)}, status=500)


class NeuralNetwork(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: List[int], 
                 output_size: int, 
                 task_type: str = 'classification',
                 dropout_rate: float = 0.2):
        super(NeuralNetwork, self).__init__()
        self.task_type = task_type
        self.batch_norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layer_block = nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                self.dropout
            )
            layers.append(layer_block)
            prev_size = hidden_size
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.hidden_layers:
            if isinstance(layer[0], nn.Linear):
                nn.init.xavier_uniform_(layer[0].weight)
                nn.init.zeros_(layer[0].bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        if self.task_type == 'classification':
            if self.output_layer.out_features == 1:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x
    
@csrf_exempt
def train_model_nn(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

    try:
        # Get form data
        dataset_id = request.POST.get('dataset_id')
        target_column = request.POST.get('targetColumn')
        model_type = request.POST.get('model', 'neural_network')
        train_split = float(request.POST.get('trainTestSplit', 80)) / 100
        print(model_type)
        
        # Get the dataset
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

        except Dataset.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found.'}, status=404)

        # Load the dataset
        try:
            if dataset.status == 'processed':
                df = pd.read_csv(dataset.cleaned_file)
            else:
                df = pd.read_csv(dataset.file_path)
        except Exception as e:
            return JsonResponse({'error': f'Error reading dataset: {str(e)}'}, status=500)

        # Get all columns except target column for features
        feature_columns = [col for col in df.columns if col != target_column]

        # Validate required parameters
        if not all([dataset_id, target_column, feature_columns]):
            missing_params = []
            if not dataset_id: missing_params.append('dataset_id')
            if not target_column: missing_params.append('target_column')
            if not feature_columns: missing_params.append('feature_columns')
            return JsonResponse({
                'error': f'Missing required parameters: {", ".join(missing_params)}'
            }, status=400)

        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_split, random_state=42
        )

        # Check if we're dealing with text data
        is_text_data = X.dtypes.apply(lambda x: x == 'object').any()

        if model_type == 'naive_bayes':
            if is_text_data:
                # For text data, use TF-IDF vectorization and Multinomial NB
                # Combine all text columns into a single text
                X_text_train = X_train.astype(str).apply(lambda x: ' '.join(x), axis=1)
                X_text_test = X_test.astype(str).apply(lambda x: ' '.join(x), axis=1)
                
                # Convert text to TF-IDF features
                vectorizer = TfidfVectorizer(max_features=1000)
                X_train_tfidf = vectorizer.fit_transform(X_text_train)
                X_test_tfidf = vectorizer.transform(X_text_test)
                
                # Use Multinomial NB for text classification
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = MultinomialNB(alpha=var_smoothing)
                model.fit(X_train_tfidf, y_train)
                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }               
                y_pred = model.predict(X_test_tfidf)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    model=model,  # Trained Naive Bayes model
                    y_true=y_test,  # True labels from the test dataset
                    y_pred=y_pred,  # Predicted labels
                    X=X_test,  # Test dataset features
                    model_identifier= f'Neural_network_model_classification_{dataset_id}'
                )
                # Save the trained model as a .pkl file
                model_filename = f"Neural_network_model_classification_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Neural Network',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )

            else:
                # For numerical data, use Gaussian NB
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train_scaled, y_train)
                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }                
                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    model=model,  # Trained Naive Bayes model
                    y_true=y_test,  # True labels from the test dataset
                    y_pred=y_pred,  # Predicted labels
                    X=X_test,  # Test dataset features
                    model_identifier= f'Neural_network_model_classification_{dataset_id}'
                )
                # Save the trained model as a .pkl file
                model_filename = f"Neural_network_model_classification_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Neural Network',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )

        else:
            # Handle neural network case
            hidden_layers = request.POST.get('hiddenLayers', '8,4')
            try:
                hidden_layers = tuple(map(int, hidden_layers.split(',')))
            except Exception:
                hidden_layers = (8, 4)  # Default values if parsing fails

            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Determine if it's a classification or regression task
            unique_values = y.nunique()
            is_classification = unique_values < 10 or y.dtype == 'object'

            if is_classification:
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=1000,
                    random_state=42,
                    solver=request.POST.get('solver', 'lbfgs')
                )
                model.fit(X_train_scaled, y_train)

                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                } 

                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    model=model,  # Trained Naive Bayes model
                    y_true=y_test,  # True labels from the test dataset
                    y_pred=y_pred,  # Predicted labels
                    X=X_test,  # Test dataset features
                    model_identifier= f'Neural_network_model_classification_{dataset_id}'
                )

                # Save the trained model as a .pkl file
                model_filename = f"Neural_network_model_classification_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Neural Network',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )

            else:
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=1000,
                    random_state=42,
                    solver=request.POST.get('solver', 'lbfgs')
                )
                model.fit(X_train_scaled, y_train)

                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }
                               
                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='regression',
                    model=model,  # Trained regression model
                    y_true=y_test,  # True target values
                    y_pred=y_pred,  # Predicted values from the model
                    X=X_test,  # Features used for predictions
                    model_identifier= f'Neural_network_model_regression_{dataset_id}'
                )
                # Save the trained model as a .pkl file
                model_filename = f"Neural_network_model_regression_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Neural Network',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='mean_squared_error',
                    metric_value=results['metrics']['mean_squared_error'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )

        return JsonResponse({
            'success': True,
            'results': results
        })

    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        return JsonResponse({
            'error': f'An error occurred during training: {str(e)}'
        }, status=500)


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np
import time


# Model Training View
def train_model(request):
    if request.method == 'POST':
        try:
            # Fetch form data
            dataset_id = request.POST.get('dataset_id')
            target_column = request.POST.get('targetColumn')
            model_name = request.POST.get('model')
            train_test_split_ratio = int(request.POST.get('trainTestSplit', 80)) / 100

            # Fetch dataset
            dataset = get_object_or_404(Dataset, id=dataset_id)

            if dataset.status != 'processed':
                return JsonResponse({'error': 'Dataset not Preprocessed. Please preprocess the dataset first.'}, status=400)

            # Load dataset
            df = pd.read_csv(dataset.cleaned_file.path)

            # Validate target column
            if target_column not in df.columns:
                return JsonResponse({'error': f"Target column '{target_column}' does not exist in the dataset."}, status=400)

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            feature_columns = [col for col in df.columns if col != target_column]

            try:
                class_counts = y.value_counts()
                print(f"Class distribution in target: {class_counts.to_dict()}")

                # Apply SMOTE if classes are imbalanced
                if class_counts.min() / class_counts.max() < 0.8:  # Threshold for imbalance
                    print("Imbalanced classes detected. Applying SMOTE to balance classes...")
                    smote = SMOTE(random_state=42)
                    X, y = smote.fit_resample(X, y)
                    print("Classes balanced using SMOTE.")
                else:
                    print("Classes are already balanced.")
            except Exception as e: 
                print("The y class is probably continuous")
                pass

            scaler = StandardScaler()

            # Convert to NumPy arrays
            # X = np.array(X)
            # y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_split_ratio, random_state=42)

            # Initialize model variable
            model = None

            # Handle model selection and parameters
            if model_name == 'linear_regression':
                # Fetch parameters for Linear Regression
                fit_intercept = request.POST.get('fitIntercept', 'true').lower() == 'true'
                model = LinearRegression(fit_intercept=fit_intercept)

                # Scale the features
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit the model on the log-transformed target
                model.fit(X_train, y_train)
                # Add feature_names_in_ attribute to the model
                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }
                y_pred = model.predict(X_test)

                # Evaluate metrics on the original scale
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='regression',
                    model=model,  # Trained regression model
                    y_true=y_test,  # True target values
                    y_pred=y_pred,  # Predicted values from the model
                    X=X_test, # Features used for predictions
                    model_identifier= f'linear_regression_model_{dataset_id}'
                )
                # print(f"Metrics: {results}")

                # Save the trained model as a .pkl file
                model_filename = f"linear_regression_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Linear Regression',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='mean_squared_error',
                    metric_value=results['metrics']['mean_squared_error'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )

                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })


            elif model_name == 'decision_tree':
                # Fetch parameters for Decision Tree
                max_depth = request.POST.get('maxDepth', 3)
                model = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)

                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }
                y_pred = model.predict(X_test)

                # Generate decision tree visualization (if applicable)
                feature_names = feature_columns  # Feature names from the dataset
                class_names = df[target_column].unique().astype(str)  # Class names based on unique target labels
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred,
                    model=model,
                    feature_names=feature_names,
                    class_names=class_names,
                    X=X_test,
                    model_identifier= f'decision_tree_model_{dataset_id}'
                    
                )

                # Save the trained model as a .pkl file
                model_filename = f"decision_tree_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Decision Tree',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy']
                )

                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })

            elif model_name == 'svm':
                # Create unique model identifier
                import datetime
                timestamp = int(datetime.datetime.now().timestamp())
                model_identifier = f'svm_{dataset_id}_{timestamp}'

                # Scale the features
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=feature_columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=feature_columns,
                    index=X_test.index
                )

                # List of kernels to try
                kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                kernel_scores = {}
                kernel_predictions = {}
                best_accuracy = 0
                best_kernel = None
                best_model = None

                # Train and evaluate SVM with different kernels
                for kernel in kernels:
                    try:
                        # Train SVM with current kernel
                        svm_model = SVC(kernel=kernel, probability=True)
                        svm_model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = svm_model.predict(X_test_scaled)
                        
                        # Calculate accuracy
                        accuracy = accuracy_score(y_test, y_pred)
                        kernel_scores[kernel] = accuracy
                        kernel_predictions[kernel] = y_pred

                        # Keep track of best performing kernel
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_kernel = kernel
                            best_model = svm_model

                    except Exception as e:
                        print(f"Error training SVM with {kernel} kernel: {str(e)}")
                        kernel_scores[kernel] = 0

                # Use the best model for final predictions
                model = best_model
                y_pred = kernel_predictions[best_kernel]

                # Create kernel comparison plot
                kernel_comparison_path = os.path.join(settings.MEDIA_ROOT, 'visualizations', model_identifier, 'kernel_comparison.png')
                os.makedirs(os.path.dirname(kernel_comparison_path), exist_ok=True)
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(kernel_scores.keys(), kernel_scores.values())
                plt.title('SVM Kernel Performance Comparison')
                plt.xlabel('Kernel Type')
                plt.ylabel('Accuracy Score')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom')
                
                # Highlight best kernel
                bars[list(kernel_scores.keys()).index(best_kernel)].set_color('green')
                plt.savefig(kernel_comparison_path)
                plt.close()

                # Store feature names and ranges for the best model
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }

                # Generate standard visualizations with the best model
                results = generate_visualizations(
                    model_type='classification',
                    model=model,
                    y_true=y_test,
                    y_pred=y_pred,
                    X=X_test_scaled,
                    feature_names=feature_columns,
                    model_identifier=model_identifier
                )

                # Add kernel comparison to visualizations
                results['visualizations']['kernel_comparison'] = f'/visualizations/{model_identifier}/kernel_comparison.png'

                # Save the best model
                model_filename = f"svm_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm=f'SVM (kernel={best_kernel})',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=best_accuracy,
                )
                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })



            elif model_name == 'random_forest':
                # Fetch parameters for Random Forest
                n_estimators = int(request.POST.get('nEstimators', 100))
                
                # Create unique model identifier using timestamp
                import datetime
                timestamp = int(datetime.datetime.now().timestamp())
                model_identifier = f'random_forest_{dataset_id}_{timestamp}'

                # Initialize and fit the model
                model = RandomForestClassifier(n_estimators=n_estimators)
                
                # Scale the features while preserving feature names
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=feature_columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=feature_columns,
                    index=X_test.index
                )

                # Fit the model with scaled data
                model.fit(X_train_scaled, y_train)

                # Store feature names and ranges
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }

                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Generate visualizations
                results = generate_visualizations(
                    model_type='classification',
                    model=model,
                    y_true=y_test,
                    y_pred=y_pred,
                    X=X_test_scaled,  # Pass scaled data with feature names
                    feature_names=feature_columns,
                    model_identifier=model_identifier
                )

                # Save the trained model
                model_filename = f"random_forest_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm=f'Random Forest (n_estimators={n_estimators})',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                )

                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })


            elif model_name == 'knn':
                import time
                # Scale the features
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Define a range of K values based on dataset size
                max_k = min(15, len(X_train) // 2)  # Ensure k is not larger than half the training set
                k_range = range(1, max_k + 1)
                
                # Perform k-fold cross validation for each k value
                cv_scores = []
                train_scores = []
                
                for k in k_range:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    # Get cross-validation scores
                    try:
                        # Use smaller number of folds for small datasets
                        n_folds = min(5, len(X_train) // 2)
                        cv_score = cross_val_score(knn, X_train, y_train, cv=n_folds, scoring='accuracy').mean()
                        cv_scores.append(cv_score)
                        
                        # Get training scores
                        knn.fit(X_train, y_train)
                        train_score = knn.score(X_train, y_train)
                        train_scores.append(train_score)
                    except ValueError as e:
                        print(f"Error with k={k}: {str(e)}")
                        continue
                
                # Find optimal k that balances performance and overfitting
                if len(cv_scores) > 0:
                    score_diff = np.array(train_scores) - np.array(cv_scores)
                    optimal_k_idx = np.argmin(score_diff)
                    optimal_k = k_range[optimal_k_idx]
                else:
                    optimal_k = 1  # fallback to 1 if no valid k found
                
                print(f"Selected optimal k: {optimal_k}")
                
                # Train final model with optimal k
                model = KNeighborsClassifier(n_neighbors=optimal_k)
                model.fit(X_train, y_train)
                
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }
                
                # Get predictions for both training and test sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Create a unique model identifier
                model_identifier = f"knn_{dataset_id}_{int(time.time())}"

                # Create K selection plot
                plt.figure(figsize=(10, 6))
                plt.plot(k_range[:len(train_scores)], train_scores, label='Training Accuracy')
                plt.plot(k_range[:len(cv_scores)], cv_scores, label='Cross-validation Accuracy')
                plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                plt.xlabel('k value')
                plt.ylabel('Accuracy')
                plt.title('K Selection Analysis')
                plt.legend()
                k_selection_path = os.path.join(settings.MEDIA_ROOT, 'visualizations', model_identifier, 'k_selection.png')
                os.makedirs(os.path.dirname(k_selection_path), exist_ok=True)
                plt.savefig(k_selection_path)
                plt.close()

                # Generate standard classification visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_test_pred,  # Use test predictions
                    model=model,
                    X=X_test,  # Use test data
                    feature_names=feature_columns,
                    model_identifier=model_identifier
                )

                # Add K selection path to results
                if 'visualizations' not in results:
                    results['visualizations'] = {}
                results['visualizations']['k_selection'] = k_selection_path
                                
                # Save the trained model
                model_filename = f"knn_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)
                
                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm=f'KNN (k={optimal_k})',
                    training_status='completed',
                    model_path=model_path
                )
                
                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                )

                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })

            elif model_name == 'polynomial_regression':
                # Create unique model identifier
                import datetime
                timestamp = int(datetime.datetime.now().timestamp())
                model_identifier = f'polynomial_regression_{dataset_id}_{timestamp}'

                # Scale the features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Test different polynomial degrees
                max_degree = 10  # Maximum degree to test
                degrees = range(1, max_degree + 1)
                degree_scores = {}
                degree_predictions = {}
                degree_models = {}
                best_r2 = float('-inf')
                best_degree = None
                best_model = None
                best_y_pred = None

                # Store MSE and R2 scores for each degree
                mse_scores = []
                r2_scores = []

                for degree in degrees:
                    try:
                        # Create polynomial features
                        poly = PolynomialFeatures(degree=degree)
                        X_train_poly = poly.fit_transform(X_train_scaled)
                        X_test_poly = poly.transform(X_test_scaled)

                        # Train model
                        lr_model = LinearRegression()
                        lr_model.fit(X_train_poly, y_train)
                        
                        # Make predictions
                        y_pred = lr_model.predict(X_test_poly)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Store scores and predictions
                        degree_scores[degree] = {'mse': mse, 'r2': r2}
                        degree_predictions[degree] = y_pred
                        degree_models[degree] = (lr_model, poly)
                        
                        mse_scores.append(mse)
                        r2_scores.append(r2)

                        # Update best model if this one is better
                        if r2 > best_r2:
                            best_r2 = r2
                            best_degree = degree
                            best_model = lr_model
                            best_y_pred = y_pred

                    except Exception as e:
                        print(f"Error training polynomial regression with degree {degree}: {str(e)}")
                        degree_scores[degree] = {'mse': float('inf'), 'r2': float('-inf')}

                # Create degree comparison plot
                comparison_path = os.path.join(settings.MEDIA_ROOT, 'visualizations', 
                                            model_identifier, 'degree_comparison.png')
                os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot MSE scores
                ax1.plot(degrees, mse_scores, 'bo-')
                ax1.axvline(x=best_degree, color='r', linestyle='--', 
                            label=f'Best degree={best_degree}')
                ax1.set_title('Mean Squared Error vs Polynomial Degree')
                ax1.set_xlabel('Polynomial Degree')
                ax1.set_ylabel('Mean Squared Error')
                ax1.grid(True)
                ax1.legend()

                # Plot R² scores
                ax2.plot(degrees, r2_scores, 'go-')
                ax2.axvline(x=best_degree, color='r', linestyle='--', 
                            label=f'Best degree={best_degree}')
                ax2.set_title('R² Score vs Polynomial Degree')
                ax2.set_xlabel('Polynomial Degree')
                ax2.set_ylabel('R² Score')
                ax2.grid(True)
                ax2.legend()

                plt.tight_layout()
                plt.savefig(comparison_path)
                plt.close()

                # Use the best model and its polynomial transformer
                model, poly = degree_models[best_degree]
                y_pred = degree_predictions[best_degree]

                # Store feature names and ranges
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }

                # Generate visualizations
                results = generate_visualizations(
                    model_type='regression',
                    model=model,
                    y_true=y_test,
                    y_pred=y_pred,
                    X=X_test_scaled,
                    feature_names=feature_columns,
                    model_identifier=model_identifier
                )

                # Add degree comparison to visualizations
                results['visualizations']['degree_comparison'] = f'/visualizations/{model_identifier}/degree_comparison.png'

                # Save both the model and polynomial transformer
                model_filename = f"polynomial_regression_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                
                # Save as a tuple of (model, poly_transformer)
                with open(model_path, 'wb') as f:
                    joblib.dump((model, poly), f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm=f'Polynomial Regression (degree={best_degree})',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='r2_score',
                    metric_value=best_r2,
                )

                # Add polynomial-specific metrics to results
                results['metrics'].update({
                    'best_degree': best_degree,
                    'best_r2_score': best_r2,
                    'degree_comparison': degree_scores
                })
                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })
            elif model_name == 'logistic_regression':
                solver = request.POST.get('solver', 'lbfgs')
                max_iter = int(request.POST.get('maxIter', 1000))
                model = LogisticRegression(solver=solver, max_iter=max_iter)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)

                model.feature_names_in_ = feature_columns  # Add feature names
                model.suggested_ranges = {  # Add suggested input ranges
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }
                y_pred = model.predict(X_test)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    model=model,  # Trained Naive Bayes model
                    y_true=y_test,  # True labels from the test dataset
                    y_pred=y_pred,  # Predicted labels
                    X=X_test,  # Test dataset features
                    model_identifier= f'logistic_regression_model_{dataset_id}'
                )

                # Save the trained model as a .pkl file
                model_filename = f"logistic_regression_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Logistic Regression',
                    training_status='completed',
                    model_path = model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=results['metrics']['accuracy'],
                    # visualization_path=f'data:image/png;base64,{image_base64}',
                )
                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })

            elif model_name == 'naive_bayes':
                # Create unique model identifier
                import datetime
                timestamp = int(datetime.datetime.now().timestamp())
                model_identifier = f'naive_bayes_{dataset_id}_{timestamp}'

                # Scale the features
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=feature_columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=feature_columns,
                    index=X_test.index
                )

                # Define range of var_smoothing values to test (logarithmic scale)
                var_smoothing_values = np.logspace(-12, 0, 13)  # Test from 1e-12 to 1e0
                smoothing_scores = {}
                smoothing_predictions = {}
                best_accuracy = 0
                best_var_smoothing = None
                best_model = None

                # Train and evaluate Naive Bayes with different var_smoothing values
                for var_smoothing in var_smoothing_values:
                    try:
                        # Train Naive Bayes with current var_smoothing
                        nb_model = GaussianNB(var_smoothing=var_smoothing)
                        nb_model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = nb_model.predict(X_test_scaled)
                        
                        # Calculate accuracy
                        accuracy = accuracy_score(y_test, y_pred)
                        smoothing_scores[var_smoothing] = accuracy
                        smoothing_predictions[var_smoothing] = y_pred

                        # Keep track of best performing var_smoothing
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_var_smoothing = var_smoothing
                            best_model = nb_model

                    except Exception as e:
                        print(f"Error training Naive Bayes with var_smoothing={var_smoothing}: {str(e)}")
                        smoothing_scores[var_smoothing] = 0

                # Use the best model for final predictions
                model = best_model
                y_pred = smoothing_predictions[best_var_smoothing]

                # Create var_smoothing comparison plot
                smoothing_comparison_path = os.path.join(settings.MEDIA_ROOT, 'visualizations', 
                                                    model_identifier, 'smoothing_comparison.png')
                os.makedirs(os.path.dirname(smoothing_comparison_path), exist_ok=True)
                
                plt.figure(figsize=(12, 6))
                plt.semilogx(list(smoothing_scores.keys()), list(smoothing_scores.values()), 'bo-')
                plt.axvline(x=best_var_smoothing, color='r', linestyle='--', 
                            label=f'Best var_smoothing={best_var_smoothing:.2e}')
                plt.title('Naive Bayes Variance Smoothing Parameter Optimization')
                plt.xlabel('Variance Smoothing Parameter (log scale)')
                plt.ylabel('Accuracy Score')
                plt.grid(True)
                plt.legend()
                
                # Add value labels for best point
                plt.plot(best_var_smoothing, best_accuracy, 'ro')
                plt.annotate(f'Best Accuracy: {best_accuracy:.3f}',
                            xy=(best_var_smoothing, best_accuracy),
                            xytext=(10, 10), textcoords='offset points')
                
                plt.savefig(smoothing_comparison_path)
                plt.close()

                # Store feature names and ranges for the best model
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min()} - {X[feature].max()}" for feature in feature_columns
                }

                # Generate standard visualizations with the best model
                results = generate_visualizations(
                    model_type='classification',
                    model=model,
                    y_true=y_test,
                    y_pred=y_pred,
                    X=X_test_scaled,
                    feature_names=feature_columns,
                    model_identifier=model_identifier
                )

                # Add smoothing comparison to visualizations
                results['visualizations']['smoothing_comparison'] = f'/visualizations/{model_identifier}/smoothing_comparison.png'

                # Save the best model
                model_filename = f"naive_bayes_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm=f'Naive Bayes (var_smoothing={best_var_smoothing:.2e})',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='accuracy',
                    metric_value=best_accuracy,
                )
                return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
                })


            elif model_name == 'kmeans':
                X = df
                feature_columns = [col for col in df.columns]

                # Ensure feature scaling with StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Elbow method to determine the optimal number of clusters
                distortions = []
                silhouette_scores = []
                k_range = range(2, 11)  # Start from 2 for silhouette score calculation

                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Calculate both metrics
                    distortions.append(kmeans.inertia_)
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    print(f"k={k}, silhouette={silhouette_avg:.3f}, inertia={kmeans.inertia_:.2f}")

                # Calculate the elbow point using the kneedle algorithm
                kn = KneeLocator(list(k_range), distortions, curve='convex', direction='decreasing')
                elbow_k = kn.elbow if kn.elbow else k_range[0]

                # Find k with maximum silhouette score
                silhouette_k = k_range[np.argmax(silhouette_scores)]

                # Combine both metrics to choose optimal k
                # Normalize both metrics to [0,1] range for fair comparison
                normalized_distortions = (max(distortions) - np.array(distortions)) / (max(distortions) - min(distortions))
                normalized_silhouette = (np.array(silhouette_scores) - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores))

                # Calculate combined score for each k
                combined_scores = []
                for i, k in enumerate(k_range):
                    combined_score = (normalized_silhouette[i] + normalized_distortions[i]) / 2
                    combined_scores.append(combined_score)

                # Choose k with highest combined score
                optimal_k = k_range[np.argmax(combined_scores)]

                print(f"Elbow method suggests k={elbow_k}")
                print(f"Silhouette analysis suggests k={silhouette_k}")
                print(f"Combined analysis suggests optimal_k={optimal_k}")

                # Visualize both metrics
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Plot elbow curve
                ax1.plot(k_range, distortions, marker='o')
                ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Distortion (Inertia)')
                ax1.set_title('Elbow Method')
                ax1.legend()

                # Plot silhouette scores
                ax2.plot(k_range, silhouette_scores, marker='o')
                ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Analysis')
                ax2.legend()

                plt.tight_layout()
                plt.show()

                # Train-test split on scaled features
                X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

                # Initialize and train the KMeans model with optimal k
                model = KMeans(
                    n_clusters=optimal_k,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )

                # Fit on the entire scaled dataset
                model.fit(X_scaled)

                # Store feature names and ranges
                model.feature_names_in_ = feature_columns
                model.suggested_ranges = {
                    feature: f"{X[feature].min():.2f} - {X[feature].max():.2f}"
                    for feature in feature_columns
                }

                # Get predictions for both sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate clustering
                try:
                    # Calculate silhouette scores on the full dataset
                    silhouette_full = silhouette_score(X_scaled, model.labels_)
                    silhouette_train = silhouette_score(X_train, y_train_pred)
                    silhouette_test = silhouette_score(X_test, y_test_pred)
                    print(f"Final Silhouette scores - Full: {silhouette_full:.3f}, Train: {silhouette_train:.3f}, Test: {silhouette_test:.3f}")
                except ValueError as e:
                    return JsonResponse({'error': f"Silhouette score calculation failed: {str(e)}"}, status=400)

                # Visualization: Reduce to 2D using PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)  # Transform entire dataset
                cluster_centers_pca = pca.transform(model.cluster_centers_)

                # Prepare elbow data for visualization
                elbow_data = {
                    'k_range': list(k_range),
                    'distortions': distortions,
                    'silhouette_scores': silhouette_scores,
                    'optimal_k': optimal_k,
                }

                # Call generate_visualizations with full dataset visualization
                results = generate_visualizations(
                    model_type='clustering',
                    model=model,
                    X=X_scaled,  # Pass full scaled dataset
                    clusters=model.labels_,  # Use full dataset labels
                    pca_data=X_pca,
                    cluster_centers_pca=cluster_centers_pca,
                    inertia=model.inertia_,
                    n_clusters=optimal_k,
                    silhouette_score=silhouette_full,  # Use full dataset silhouette score
                    model_identifier=f'kmeans_model_{dataset_id}',
                    elbow_data=elbow_data
                )
                print(f"Final optimal_k: {optimal_k}, Inertia: {model.inertia_:.2f}")


            else:
                return JsonResponse({'error': 'Invalid model selected.'}, status=400)

            return JsonResponse({
                    'success': True,
                    'model_id': ml_model.id,  # Add this line
                    'results': results
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def generate_visualizations(
        model_type, y_true=None, y_pred=None, model=None, 
        feature_names=None, model_identifier=None,
        class_names=None, X=None, clusters=None, 
        n_clusters=None, cluster_centers_pca=None, 
        pca_data=None, silhouette_score=None, inertia=None,
        elbow_data=None, knn_data=None):  
    
    # Create a unique directory for the model's visualizations
    model_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations', model_identifier or str(uuid.uuid4()))
    os.makedirs(model_dir, exist_ok=True)
    visualizations = {}
    additional_metrics = {}

    if model_type == 'classification':
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_path, format='png', bbox_inches='tight')
        visualizations['confusion_matrix'] = cm_path
        plt.close()

        # Add SVM-specific visualizations
        if isinstance(model, SVC) and X is not None and X.shape[1] >= 2:
            # Create decision boundary visualization if possible
            decision_boundary_path = os.path.join(model_dir, 'decision_boundary.png')
            
            # Use PCA for high-dimensional data
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                
                # Train a new SVM model on the 2D data for visualization
                svm_2d = SVC(kernel=model.kernel)
                svm_2d.fit(X_2d, y_true)
                
                # Create meshgrid
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))
                
                # Make predictions using the 2D model
                Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                plt.figure(figsize=(10, 8))
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, alpha=0.8)
                plt.title(f'SVM Decision Boundary (kernel={model.kernel}) - PCA Visualization')
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                plt.savefig(decision_boundary_path, format='png', bbox_inches='tight')
                visualizations['decision_boundary'] = decision_boundary_path
                plt.close()
            else:
                # For 2D data, use original features
                # Create meshgrid
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))
                
                # Make predictions
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                plt.figure(figsize=(10, 8))
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X[:, 0], X[:, 1], c=y_true, alpha=0.8)
                plt.title(f'SVM Decision Boundary (kernel={model.kernel})')
                plt.xlabel(feature_names[0] if feature_names else 'Feature 1')
                plt.ylabel(feature_names[1] if feature_names else 'Feature 2')
                plt.savefig(decision_boundary_path, format='png', bbox_inches='tight')
                visualizations['decision_boundary'] = decision_boundary_path
                plt.close()


        # Generate decision tree visualization (if applicable)
        if model and isinstance(model, DecisionTreeClassifier):
            tree_path = os.path.join(model_dir, 'decision_tree.png')
            plt.figure(figsize=(20, 12))
            plot_tree(
                model,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=10
            )
            plt.title('Decision Tree Visualization')
            plt.savefig(tree_path, format='png', bbox_inches='tight')
            visualizations['decision_tree'] = tree_path
            plt.close()

        # Add Random Forest-specific visualizations
        if isinstance(model, RandomForestClassifier):
            # Feature Importance Plot (Enhanced)
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            feature_importance_path = os.path.join(model_dir, 'feature_importance.png')
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances (Random Forest)")
            plt.bar(range(X.shape[1]), importances[indices],
                   color="b", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(feature_importance_path, format='png', bbox_inches='tight')
            visualizations['feature_importance'] = feature_importance_path
            plt.close()

            # Tree Structure Visualization (sample tree)
            sample_tree_path = os.path.join(model_dir, 'sample_tree.png')
            plt.figure(figsize=(20, 10))
            plot_tree(model.estimators_[0],
                     feature_names=feature_names,
                     filled=True,
                     rounded=True,
                     fontsize=10)
            plt.title("Sample Decision Tree from Random Forest")
            plt.savefig(sample_tree_path, format='png', bbox_inches='tight')
            visualizations['sample_tree'] = sample_tree_path
            plt.close()

            # OOB Error Plot (if available)
            if hasattr(model, 'oob_score_'):
                oob_path = os.path.join(model_dir, 'oob_error.png')
                plt.figure(figsize=(10, 6))
                plt.title("Out-of-Bag Error Rate")
                plt.plot(range(1, model.n_estimators + 1), 
                        [1 - model.oob_score_] * model.n_estimators,
                        'r--', label='OOB Error Rate')
                plt.xlabel("Number of Trees")
                plt.ylabel("Error Rate")
                plt.legend()
                plt.savefig(oob_path, format='png', bbox_inches='tight')
                visualizations['oob_error'] = oob_path
                plt.close()

            # Feature Importance Heatmap
            importance_heatmap_path = os.path.join(model_dir, 'importance_heatmap.png')
            plt.figure(figsize=(12, 8))
            importance_matrix = np.array([tree.feature_importances_ for tree in model.estimators_])
            sns.heatmap(importance_matrix, xticklabels=feature_names, 
                       yticklabels=False, cmap='YlOrRd')
            plt.title("Feature Importance Across Trees")
            plt.xlabel("Features")
            plt.ylabel("Trees")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(importance_heatmap_path, format='png', bbox_inches='tight')
            visualizations['importance_heatmap'] = importance_heatmap_path
            plt.close()
        # Add KNN-specific visualizations
        if isinstance(model, KNeighborsClassifier) and knn_data is not None:
            # K Selection Analysis
            k_selection_path = os.path.join(model_dir, 'k_selection.png')
            plt.figure(figsize=(12, 6))
            
            if len(knn_data['cv_scores']) > 0:
                plt.plot(knn_data['k_range'][:len(knn_data['train_scores'])], 
                        knn_data['train_scores'], 
                        label='Training Accuracy')
                plt.plot(knn_data['k_range'][:len(knn_data['cv_scores'])], 
                        knn_data['cv_scores'], 
                        label='Cross-validation Accuracy')
                plt.axvline(x=knn_data['optimal_k'], color='r', 
                          linestyle='--', 
                          label=f'Optimal k={knn_data["optimal_k"]}')
            
            plt.xlabel('k value')
            plt.ylabel('Accuracy')
            plt.title('K Selection Analysis')
            plt.legend()
            plt.savefig(k_selection_path, format='png', bbox_inches='tight')
            visualizations['k_selection'] = k_selection_path
            plt.close()

            # Decision Boundary (if applicable)
            if X is not None and X.shape[1] >= 2:
                decision_boundary_path = os.path.join(model_dir, 'decision_boundary.png')
                
                # Create PCA if more than 2 dimensions
                if X.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X)
                else:
                    X_2d = X
                
                # Create meshgrid
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))
                
                # Fit a new KNN model on 2D data
                knn_2d = KNeighborsClassifier(n_neighbors=knn_data['optimal_k'])
                knn_2d.fit(X_2d, y_true)
                
                # Make predictions
                Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                plt.figure(figsize=(10, 8))
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, alpha=0.8)
                plt.title('KNN Decision Boundary')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.savefig(decision_boundary_path, format='png', bbox_inches='tight')
                visualizations['decision_boundary'] = decision_boundary_path
                plt.close()

        # ROC Curve
        if hasattr(model, 'predict_proba') and y_true is not None and y_pred is not None:
            y_score = model.predict_proba(X)[:, 1]  # Probability scores for the positive class
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            roc_path = os.path.join(model_dir, 'roc_curve.png')
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.savefig(roc_path, format='png', bbox_inches='tight')
            visualizations['roc_curve'] = roc_path
            plt.close()

        # Precision-Recall Curve
        if hasattr(model, 'predict_proba') and y_true is not None and y_pred is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)

            pr_path = os.path.join(model_dir, 'precision_recall_curve.png')
            plt.figure(figsize=(10, 6))
            plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            plt.savefig(pr_path, format='png', bbox_inches='tight')
            visualizations['precision_recall_curve'] = pr_path
            plt.close()

        # Feature Importance Plot
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            importances = model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            sorted_features = np.array(feature_names)[sorted_indices]
            sorted_importances = importances[sorted_indices]

            feature_importance_path = os.path.join(model_dir, 'feature_importance.png')
            plt.figure(figsize=(12, 8))
            sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
            plt.title('Feature Importance Plot')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.savefig(feature_importance_path, format='png', bbox_inches='tight')
            visualizations['feature_importance'] = feature_importance_path
            plt.close()

        # Learning Curve
        if X is not None and y_true is not None:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y_true, cv=5, scoring='accuracy', n_jobs=-1
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            learning_curve_path = os.path.join(model_dir, 'learning_curve.png')
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
            plt.plot(train_sizes, test_mean, label='Cross-Validation Score', color='green')
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')
            plt.title('Learning Curve')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy Score')
            plt.legend(loc='lower right')
            plt.savefig(learning_curve_path, format='png', bbox_inches='tight')
            visualizations['learning_curve'] = learning_curve_path
            plt.close()

        # Accuracy metric
        accuracy = accuracy_score(y_true, y_pred)
        additional_metrics['accuracy'] = round(accuracy, 3)


    elif model_type == 'regression':
        # Scatter Plot: Actual vs Predicted
        lin_path = os.path.join(model_dir, 'linear_graph.png')
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, label='Predicted vs Actual')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Ideal Fit')
        plt.title('Regression Results: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.savefig(lin_path, format='png', bbox_inches='tight')
        visualizations['linear_graph'] = lin_path
        plt.close()

        # Residuals Plot
        residuals = y_true - y_pred
        residuals_path = os.path.join(model_dir, 'residuals_plot.png')
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.title('Residuals Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.savefig(residuals_path, format='png', bbox_inches='tight')
        visualizations['residuals_plot'] = residuals_path
        plt.close()

        # Histogram of Residuals
        plt.figure(figsize=(10, 6))
        histo_path = os.path.join(model_dir, 'histogram.png')
        plt.hist(residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.savefig(histo_path, format='png', bbox_inches='tight')
        visualizations['histogram_residuals'] = histo_path
        plt.close()

        # Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y_true, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        learning_curve_path = os.path.join(model_dir, 'learning_curve.png')

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-Validation Score')
        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score (R²)')
        plt.legend()
        plt.savefig(learning_curve_path, format='png', bbox_inches='tight')
        visualizations['learning_curve'] = learning_curve_path
        plt.close()

        # Metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        additional_metrics = {
            'mean_squared_error': round(mse, 3),
            'r2_score': round(r2, 3)
        }


    elif model_type == 'clustering':
        # Generate elbow curve
        if elbow_data is not None:
            elbow_plot_path = os.path.join(model_dir, 'elbow_plot.png')
            plt.figure(figsize=(10, 6))
            plt.plot(elbow_data['k_range'], elbow_data['distortions'], 'bx-')
            plt.axvline(x=elbow_data['optimal_k'], color='r', linestyle='--', 
                    label=f'Optimal k={elbow_data["optimal_k"]}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Distortion (Inertia)')
            plt.title('Elbow Method for Optimal k')
            plt.legend()
            plt.grid(True)
            plt.savefig(elbow_plot_path, format='png', bbox_inches='tight')
            visualizations['elbow_curve'] = elbow_plot_path
            plt.close()

        # PCA Visualization for Clusters
        if pca_data is not None and clusters is not None and cluster_centers_pca is not None:
            clusters_path = os.path.join(model_dir, 'clusters_plot.png')
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                hue=clusters,
                palette='viridis',
                alpha=0.7
            )
            plt.scatter(
                cluster_centers_pca[:, 0],
                cluster_centers_pca[:, 1],
                c='red',
                s=200,
                marker='X',
                label='Centroids'
            )
            plt.title(f'KMeans Clustering Visualization (k={n_clusters})')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend(title="Cluster")
            plt.savefig(clusters_path, format='png', bbox_inches='tight')
            visualizations['clusters_visual'] = clusters_path
            plt.close()
            
        # Silhouette Score Plot
        if X is not None and clusters is not None and silhouette_score is not None and n_clusters is not None:
            from sklearn.metrics import silhouette_samples

            silhouette_vals = silhouette_samples(X, clusters)
            y_lower = 0

            silhouette_path = os.path.join(model_dir, 'silhouette_score.png')
            plt.figure(figsize=(10, 6))
            for i in range(n_clusters):
                cluster_silhouette_vals = silhouette_vals[clusters == i]
                cluster_silhouette_vals.sort()
                y_upper = y_lower + len(cluster_silhouette_vals)
                plt.barh(
                    range(y_lower, y_upper),
                    cluster_silhouette_vals,
                    height=1.0,
                    edgecolor='none',
                    label=f"Cluster {i + 1}"
                )
                y_lower += len(cluster_silhouette_vals)

            plt.axvline(silhouette_score, color='red', linestyle='--', label='Average Silhouette Score')
            plt.xlabel('Silhouette Coefficient Values')
            plt.ylabel('Cluster')
            plt.title('Silhouette Plot')
            plt.legend()
            plt.savefig(silhouette_path, format='png', bbox_inches='tight')
            visualizations['silhouette_plot'] = silhouette_path
            plt.close()

        # Additional Clustering Metrics
        if inertia is not None and silhouette_score is not None:
            additional_metrics = {
                'inertia': inertia,
                'silhouette_score': silhouette_score
            }



    return {
        'visualizations': visualizations,
        'metrics': additional_metrics,
        'model_type': model_type
    }


def training_page(request,id):

    try:
        dataset = Dataset.objects.get(id=id)
        dataset_id = id

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    return render(request, 'model_training.html', {'dataset': dataset, 'dataset_id' : dataset_id})

def get_columns(request):
    if request.method == 'GET':
        dataset_id = request.GET.get('datasetId')

        # Assuming you have a Dataset model and a method to fetch the dataset file
        dataset = Dataset.objects.get(id=dataset_id)
        if dataset.status == 'processed':
            dataset_path = dataset.cleaned_file.path  # Path to the dataset file
        else:            
            dataset_path = dataset.file_path.path

        try:
            # Load the dataset (e.g., CSV file)
            data = pd.read_csv(dataset_path)
            columns = list(data.columns)  # Get the list of columns
            return JsonResponse({'columns': columns})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

def render_predictions_view(request):
    # Fetch models to display in the dropdown
    models = MLModel.objects.filter(training_status='completed').order_by('-created_at')
    return render(request, 'predictions.html', {
        'models': models,
        'target_columns': None,  # Initial state; no dataset uploaded
        'results': None,         # No predictions yet
        'metrics': None,         # No metrics yet
    })

# def fetch_models(request):
#     models = MLModel.objects.all().values('id', 'algorithm', 'created_at')
#     return JsonResponse({'models': list(models)})

def fetch_models(request):
    # Fetch all models with their related dataset and results
    models = MLModel.objects.select_related('dataset').prefetch_related('results').all()
    model_list = []
    
    for model in models:
        # Get accuracy from ModelResult if it exists
        accuracy_result = ModelResult.objects.filter(
            model=model,
            metric_name='accuracy'
        ).first()
        
        accuracy_value = accuracy_result.metric_value if accuracy_result else None
        dataset_name = model.dataset.name if model.dataset else 'Unknown Dataset'
        
        model_data = {
            'id': model.id,
            'algorithm': model.algorithm,
            'created_at': model.created_at,
            'dataset_id': model.dataset.id if model.dataset else None,
            'dataset_name': dataset_name,
            'accuracy': accuracy_value
        }
        model_list.append(model_data)
    
    return JsonResponse({'models': model_list})

def fetch_visualizations(request):
    model_id = request.GET.get('id')
    if not model_id:
        return JsonResponse({'error': 'No model ID provided'}, status=400)
    
    try:
        model = MLModel.objects.get(id=model_id)
        model_path = os.path.splitext(os.path.basename(model.model_path.path))[0]
        visualizations_dir = os.path.join(settings.MEDIA_ROOT, "visualizations", model_path)
        
        if not os.path.exists(visualizations_dir):
            return JsonResponse({"visualizations": []})
        
        visualizations = []
        for file in os.listdir(visualizations_dir):
            if file.endswith('.png'):
                # Create a title from the filename
                title = ' '.join(file.split('.')[0].split('_')).title()
                
                # Get description based on the type of visualization
                description = None
                if 'confusion_matrix' in file.lower():
                    description = 'Shows the model\'s prediction accuracy across different classes'
                elif 'roc_curve' in file.lower():
                    description = 'Receiver Operating Characteristic curve showing true vs false positive rates'
                elif 'learning_curve' in file.lower():
                    description = 'Model\'s learning progress during training'
                elif 'feature_importance' in file.lower():
                    description = 'Relative importance of different features in making predictions'
                elif 'precision_recall' in file.lower():
                    description = 'Trade-off between precision and recall at different thresholds'
                
                visualizations.append({
                    'title': title,
                    'url': f'/media/visualizations/{model_path}/{file}',
                    'description': description
                })
        
        return JsonResponse({"visualizations": visualizations})
    except MLModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def model_visualizations(request):
    model_id = request.GET.get('id')
    if not model_id:
        return redirect('predictions')
        
    try:
        model = MLModel.objects.get(id=model_id)
        context = {
            'model': model,
            'page_title': f'Visualizations for {model.algorithm} Model'
        }
        return render(request, 'model_visualizations.html', context)
    except MLModel.DoesNotExist:
        messages.error(request, 'Model not found')
        return redirect('predictions')

def model_predictions(request):
    return render(request, 'model_predictions.html')

def download_visualizations(request):
    model_id = request.GET.get('id')
    if not model_id:
        return HttpResponse("Model ID not provided.", status=400)

    model = MLModel.objects.get(id=model_id)
    model_path = os.path.splitext(os.path.basename(model.model_path.path))[0]
    print(f"visualizations folder name is {model_path}")
    visualizations_dir = os.path.join(settings.MEDIA_ROOT, "visualizations", model_path)
    print(f"visualizations folder directory is {visualizations_dir}")

    if not os.path.exists(visualizations_dir):
        return HttpResponse("No visualizations found.", status=404)

    zip_filename = f"visualizations_model_{model_id}.zip"
    zip_path = os.path.join(settings.MEDIA_ROOT, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(visualizations_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, visualizations_dir))

    # Serve the ZIP file
    response = HttpResponse(open(zip_path, 'rb'), content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="{zip_filename}"'

    # Optional: Clean up the ZIP file after serving
    os.remove(zip_path)

    return response

def fetch_model_details(request):
    model_id = request.GET.get('id')
    if not model_id:
        return JsonResponse({"error": "Model ID not provided"}, status=400)

    try:
        # Fetch the ML model from the database
        ml_model = MLModel.objects.get(id=model_id)
        model = joblib.load(ml_model.model_path.path)
        accuracy_result = ModelResult.objects.filter(
            model=ml_model,
            metric_name='accuracy'
        ).first()
        
        accuracy_value = accuracy_result.metric_value if accuracy_result else None
        # dataset_name = ml_model.dataset.name if ml_model.dataset else 'Unknown Dataset'
        print(accuracy_value)

        # Ensure the model contains feature_names_in_ and suggested_ranges
        if not hasattr(model, 'feature_names_in_') or not hasattr(model, 'suggested_ranges'):
            return JsonResponse({"error": "Model does not contain feature metadata. Retrain the model with updated code."}, status=500)

        # Prepare feature details
        features = [
            {
                "name": feature_name,
                "suggested_range": model.suggested_ranges.get(feature_name, "Unknown")
            }
            for feature_name in model.feature_names_in_
        ]

        return JsonResponse({
            "algorithm": ml_model.algorithm,
            "created_at": ml_model.created_at.strftime("%Y-%m-%d"),
            "features": features,
            "accuracy_value" : accuracy_value,
            "dataset_name" : ml_model.dataset.name,
            "dataset_id" : ml_model.dataset.id,
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def make_prediction(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_id = request.GET.get('id')
    if not model_id:
        return JsonResponse({"error": "Model ID not provided"}, status=400)

    try:
        # Fetch the ML model from the database
        ml_model = MLModel.objects.get(id=model_id)
        model = joblib.load(ml_model.model_path.path)

        # Ensure the model contains feature_names_in_
        if not hasattr(model, 'feature_names_in_'):
            return JsonResponse({"error": "Model does not contain feature metadata. Retrain the model with updated code."}, status=500)

        # Parse the input data
        payload = json.loads(request.body)
        input_data = []

        for feature in model.feature_names_in_:
            if feature not in payload:
                return JsonResponse({"error": f"Missing input for feature: {feature}"}, status=400)
            input_data.append(payload[feature])

        # Make prediction
        prediction = model.predict([input_data])[0]

        # Convert prediction to a standard Python data type
        if isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()  # Converts to Python int or float

        return JsonResponse({"prediction": prediction})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



@csrf_exempt
def perform_predictions(request):
    models = MLModel.objects.filter(training_status='completed').order_by('-created_at')
    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            # Handle dataset upload
            uploaded_file = request.FILES['dataset']
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Read the dataset based on the file type
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'})
            
            dataset = clean_dataset(df, delete_header=False)

            # Extract target column and model information
            model_id = request.POST['model']
            model_entry = models.get(id=model_id)
            model_path = model_entry.model_path.path
            model_algorithm = model_entry.algorithm

            if not os.path.exists(model_path):
                return JsonResponse({'error': 'Model file not found.'})

            # Load the trained model
            model = joblib.load(model_path)

            # Prepare features
            X = dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Check if the model is KMeans
            if model_algorithm == 'KMeans':
                # Predict cluster labels
                cluster_labels = model.predict(X_scaled)

                # Append cluster labels to dataset
                dataset['Cluster'] = cluster_labels

                # Evaluate clustering if silhouette score is applicable
                try:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    metrics = {
                        "type": "clustering",
                        "silhouette_score": round(silhouette, 3),
                        "inertia": round(model.inertia_, 3),
                    }
                except ValueError as e:
                    metrics = {
                        "type": "clustering",
                        "error": f"Silhouette score calculation failed: {str(e)}",
                        "inertia": round(model.inertia_, 3),
                    }

                # (Optional) PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                dataset['PCA_1'] = X_pca[:, 0]
                dataset['PCA_2'] = X_pca[:, 1]

            else:
                # Handle other models (Polynomial Regression, etc.)
                target_column = request.POST['target']
                print(f"target column is {target_column}")
                X = dataset.drop(columns=[target_column])
                y = dataset[target_column]
                X_scaled = scaler.fit_transform(X)
                if model_algorithm == 'Polynomial Regression':
                    print('starting polynomial regression')
                    poly = PolynomialFeatures(degree=2)
                    X_transformed = poly.fit_transform(X_scaled)
                else:
                    X_transformed = X_scaled

                predictions = model.predict(X_transformed)

                # Determine if it's regression or classification
                if hasattr(model, "predict_proba") or len(set(y)) <= 2:
                    accuracy = accuracy_score(y, predictions)
                    cm = confusion_matrix(y, predictions)
                    metrics = {
                        "type": "classification",
                        "accuracy": round(accuracy, 3),
                        "confusion_matrix": cm.tolist(),
                    }
                else:
                    mse = mean_squared_error(y, predictions)
                    r2 = r2_score(y, predictions)
                    metrics = {
                        "type": "regression",
                        "mse": round(mse, 3),
                        "r2": round(r2, 3),
                    }

                # Append predictions to dataset
                dataset['Predicted'] = predictions

            # Convert dataset to HTML for rendering
            # results_html = dataset.to_html(classes='table-auto w-full text-center border-collapse')

            # Return the results and metrics as JSON
            return JsonResponse({
                # 'results': results_html,
                'metrics': metrics,
            })

        except Exception as e:
            return JsonResponse({'error': str(e)})

    # Return an error if the request is invalid
    return JsonResponse({'error': 'Invalid request. Please use POST with a valid dataset.'})

@csrf_exempt
def fetch_columns(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            # Handle dataset upload
            uploaded_file = request.FILES['dataset']
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Read the dataset based on the file type
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'})
            
            # Clean dataset and extract columns
            dataset = clean_dataset(df, delete_header=False)
            columns = dataset.columns.tolist()

            # Return the list of columns as a JSON response
            return JsonResponse({'columns': columns})
        
        except Exception as e:
            return JsonResponse({'error': str(e)})

    # Return an error if the request is invalid
    return JsonResponse({'error': 'Invalid request. Please use POST with a valid dataset.'})


from django.shortcuts import render, get_object_or_404

def delete_dataset(request, dataset_id):
    if not request.user.is_authenticated:
        return redirect('signin')
    
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, 'Dataset deleted successfully.')
        return redirect('general_dashboard')
    
    return redirect('general_dashboard')



@login_required
def download_model(request):
    import logging
    logger = logging.getLogger(__name__)
    if request.method == 'GET' or request.method == 'POST':
        try:
            logger.info("Starting model download process")
            # Get the latest model for the user from the MLModel table
            latest_model = MLModel.objects.filter(
                dataset__user=request.user,
                training_status='completed'
            ).latest('created_at')
            
            logger.info(f"Found latest model: {latest_model.model_path}")
            
            if not latest_model or not latest_model.model_path:
                logger.error("No model path found")
                return JsonResponse({'error': 'No trained model found'}, status=404)
            
            model_file = latest_model.model_path.path  # Use .path to get the full filesystem path
            logger.info(f"Model file path: {model_file}")
            
            if not os.path.exists(model_file):
                logger.error(f"Model file not found at path: {model_file}")
                return JsonResponse({'error': 'Model file not found'}, status=404)
            
            # Read the model file and serve it
            try:
                with open(model_file, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='application/octet-stream')
                    filename = os.path.basename(model_file)
                    response['Content-Disposition'] = f'attachment; filename={filename}'
                    logger.info(f"Successfully prepared response with filename: {filename}")
                    return response
            except IOError as e:
                logger.error(f"Error reading model file: {str(e)}")
                return JsonResponse({'error': f'Error reading model file: {str(e)}'}, status=500)
                
        except MLModel.DoesNotExist:
            logger.error("No trained model found in database")
            return JsonResponse({'error': 'No trained model found'}, status=404)
        except Exception as e:
            logger.error(f"Unexpected error in download_model: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    logger.error("Invalid request method")
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def tutorials(request):
    tutorial_videos = [
        {
            'id': 1,
            'title': 'How to Create a Dataset',
            'video_url': 'tutorials/how to create a dataset.mp4',
            'level': 'Beginner',
            'duration': '5 min',
            'description': 'Master the fundamentals of dataset creation in DataFlowDesk.',
            'key_points': [
                'Dataset creation interface walkthrough',
                'Understanding data formats',
                'Best practices and organization'
            ]
        },
        {
            'id': 2,
            'title': 'How to Upload Dataset',
            'video_url': 'tutorials/how to ulpoad dataset.mp4',
            'level': 'Beginner',
            'duration': '3 min',
            'description': 'Learn the seamless process of uploading your datasets.',
            'key_points': [
                'Supported file formats',
                'Upload process walkthrough',
                'Troubleshooting tips'
            ]
        },
        {
            'id': 3,
            'title': 'How to Import Dataset from Kaggle',
            'video_url': 'tutorials/how to import dataset from kaggle.mp4',
            'level': 'Intermediate',
            'duration': '7 min',
            'description': 'Explore how to leverage Kaggle\'s vast dataset repository.',
            'key_points': [
                'Kaggle API setup',
                'Dataset browsing and selection',
                'Import optimization techniques'
            ]
        },
        {
            'id': 4,
            'title': 'How to Clean Dataset and Create Visualizations',
            'video_url': 'tutorials/how to clean dataset and create visualizations.mp4',
            'level': 'Advanced',
            'duration': '10 min',
            'description': 'Master advanced data cleaning and visualization techniques.',
            'key_points': [
                'Data cleaning strategies',
                'Visualization techniques',
                'Pattern interpretation'
            ]
        },
        {
            'id': 5,
            'title': 'How to Train Model and Make Predictions',
            'video_url': 'tutorials/how to train model and make predictions.mp4',
            'level': 'Expert',
            'duration': '12 min',
            'description': 'Learn advanced model training and prediction techniques.',
            'key_points': [
                'Model selection strategies',
                'Training optimization',
                'Prediction accuracy tips'
            ]
        }
    ]
    
    return render(request, 'tutorials.html', {'tutorials': tutorial_videos})

def documentation(request):
    # Quick start guide sections
    quick_start = [
        {
            'title': 'Dataset Management',
            'description': 'Upload your dataset or create a new one from scratch. Supported formats include CSV, Excel, and JSON files. You can also connect to databases or use sample datasets.',
            'code': '''# Supported file formats
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- SQL databases
- Sample datasets'''
        },
        {
            'title': 'Data Cleaning & Preprocessing',
            'description': 'Clean and prepare your data for analysis. Handle missing values, remove duplicates, normalize data, and encode categorical variables.',
            'code': '''# Available preprocessing options
1. Handle missing values
   - Remove rows
   - Fill with mean/median/mode
   - Forward/backward fill
2. Remove duplicates
3. Scale/normalize data
   - Min-Max scaling
   - Standard scaling
   - Robust scaling
4. Encode categorical variables
   - One-hot encoding
   - Label encoding
   - Ordinal encoding'''
        },
        {
            'title': 'Data Visualization',
            'description': 'Create interactive visualizations to understand your data better. Customize X and Y axes, plot types, and styling options.',
            'code': '''# Available plot types
1. Statistical plots
   - Histograms
   - Box plots
   - Violin plots
   - KDE plots
2. Relationship plots
   - Scatter plots
   - Line plots
   - Bar plots
   - Heat maps
3. Distribution plots
   - Pair plots
   - Joint plots
4. Time series plots
   - Line plots
   - Area plots
   - Seasonal decomposition'''
        },
        {
            'title': 'Model Training',
            'description': 'Train your machine learning models using various algorithms and techniques. Customize hyperparameters and validation strategies.',
            'code': '''# Available algorithms
1. Classification
   - Logistic Regression
   - Random Forest
   - Support Vector Machines
   - Gradient Boosting
   - Neural Networks
2. Regression
   - Linear Regression
   - Ridge/Lasso
   - Decision Trees
   - XGBoost
   - LightGBM
3. Clustering
   - K-Means
   - DBSCAN
   - Hierarchical Clustering
4. Dimensionality Reduction
   - PCA
   - t-SNE
   - UMAP'''
        },
        {
            'title': 'Model Evaluation & Predictions',
            'description': 'Evaluate model performance, make predictions on new data, and export results. Download trained models and visualization graphs.',
            'code': '''# Available features
1. Model evaluation metrics
   - Accuracy, Precision, Recall
   - ROC curves, AUC
   - Confusion matrix
   - R², MSE, MAE
2. Cross-validation
   - K-fold
   - Stratified K-fold
   - Time series split
3. Export options
   - Download trained model (.pkl)
   - Export predictions (.csv)
   - Save visualizations (.png, .pdf)
   - Generate reports (.html, .pdf)'''
        }
    ]

    # API Reference sections
    api_reference = [
        {
            'id': 'data-management',
            'name': 'Data Management API',
            'description': 'Upload, create, and manage datasets'
        },
        {
            'id': 'preprocessing',
            'name': 'Data Preprocessing API',
            'description': 'Clean and prepare your data'
        },
        {
            'id': 'visualization',
            'name': 'Visualization API',
            'description': 'Create and customize visualizations'
        },
        {
            'id': 'model-training',
            'name': 'Model Training API',
            'description': 'Train and tune machine learning models'
        },
        {
            'id': 'predictions',
            'name': 'Predictions API',
            'description': 'Make predictions and export results'
        }
    ]

    # Example sections
    examples = [
        {
            'id': 'classification',
            'title': 'Classification Example',
            'description': 'Build a classification model for customer churn prediction'
        },
        {
            'id': 'regression',
            'title': 'Regression Example',
            'description': 'Create a house price prediction model'
        },
        {
            'id': 'clustering',
            'title': 'Clustering Example',
            'description': 'Customer segmentation using K-means'
        },
        {
            'id': 'time-series',
            'title': 'Time Series Analysis',
            'description': 'Sales forecasting with ARIMA and Prophet'
        }
    ]

    # Resource sections
    resources = [
        {
            'title': 'Video Tutorials',
            'description': 'Step-by-step video guides for ru2yaAI',
            'url': '/tutorials/'
        },
        {
            'title': 'API Documentation',
            'description': 'Detailed API reference and examples',
            'url': '#'
        },
        {
            'title': 'Best Practices',
            'description': 'ML workflow best practices and tips',
            'url': '#'
        },
        {
            'title': 'Sample Projects',
            'description': 'Example projects and use cases',
            'url': '#'
        }
    ]

    context = {
        'quick_start': quick_start,
        'api_reference': api_reference,
        'examples': examples,
        'resources': resources
    }

    return render(request, 'documentation.html', context)
