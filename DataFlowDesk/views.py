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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import pandas as pd
import numpy as np
import json
import logging

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import logging

logger = logging.getLogger(__name__)



def index(request) :
    return render( request, "index.html")

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


def generate_plot_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')


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
        # ... existing data loading code ...
        
        # Calculate new metrics
        # 1. Missing Values
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        missing_percentage = (total_missing / (data.shape[0] * data.shape[1])) * 100
        
        # 2. Duplicate Records
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        # 3. Feature Types
        # Get the original categorical columns that were encoded
        categorical_columns = data.attrs.get('categorical_columns', [])
        
        feature_types = {
            'Numerical': len(data.select_dtypes(include=['int64', 'float64']).columns) - len(categorical_columns),
            'Categorical': len(categorical_columns),
            'DateTime': len(data.select_dtypes(include=['datetime64']).columns),
            'Boolean': len(data.select_dtypes(include=['bool']).columns)
        }
        
        # 4. Memory Usage
        memory_usage = data.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage / (1024 * 1024)  # Convert to MB

        
        # Create target class distribution plot
        plt.figure(figsize=(8, 6))
        target_column = dataset.target_class  # Get the actual target column name
        target_counts = data[target_column].value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        target_dist_plot = generate_plot_base64(plt.gcf())
        plt.close()

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

from django.forms import modelform_factory
from .models import Dataset
import csv
import os
import uuid
from django.conf import settings


import os
import csv
from datetime import datetime
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Dataset
from django.shortcuts import render, redirect
from django.urls import reverse  # For generating URLs


def create_dataset_step2(request):
    if request.method == "POST":
        try:
            # Get all the form data
            data = request.POST.getlist('data')
            dataset_meta = json.loads(request.POST.get('dataset_meta', '{}'))
            
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

import pandas as pd
from django.shortcuts import render, redirect
from .models import Dataset
import pandas as pd
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
import plotly.express as px

from django.http import JsonResponse
import pandas as pd
from .models import Dataset

from django.http import JsonResponse
import pandas as pd
from .models import Dataset



from django.http import JsonResponse
import pandas as pd
from .models import Dataset
import json

from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import json
from .models import Dataset
from django.contrib import messages

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

            # Update the Dataset table
            # dataset.cleaned_file.name = new_file_path  # Update file path
            # dataset.status = 'processed'  # Update status
            # dataset.columns_info = {  # Log normalization statistics
            #     col: {
            #         "mean": round(df[col].mean(), 2),
            #         "std_dev": round(df[col].std(), 2)
            #     } for col in numeric_columns
            # }
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

from django.shortcuts import render, get_object_or_404



import os
import io
import json
import base64
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import Dataset


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


import os
import io
import json
import base64
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import Dataset

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


from io import StringIO
from django.core.files.base import ContentFile
import uuid
import os
import pylightxl
import pandas as pd
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from .models import *
from django.core.files.storage import default_storage
from datetime import datetime
import json
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
                'preview': preview_html
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
    
    # Keep track of categorical columns
    categorical_columns = []

    # If user wants to delete header
    if delete_header:
        print("Deleting header and using first data row as new header...")
        new_headers = df.iloc[0].values.tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = new_headers
        print("Header replaced with first data row.")

    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - df.shape[0]
    print(f"Removed {removed_duplicates} duplicate rows.")

    # Process each column
    for col in df.columns:
        print(f"Processing column: {col} (Type: {df[col].dtype})")

        # Handle boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in boolean column '{col}' with 'False'.")
                df[col].fillna(False, inplace=True)  # Default strategy: replace with `False`

            # Convert boolean to numeric if needed
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

            # Encode categorical data
            print(f"Encoding categorical column '{col}' using Label Encoding.")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Handle other types
        else:
            print(f"Skipping column '{col}' as it does not fit numeric, boolean, or categorical types.")

    # Store categorical column information in the DataFrame attributes
    df.attrs['categorical_columns'] = categorical_columns
    
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
        cleaned_df.to_csv(cleaned_file_path, index=False)

        # Update the dataset object to reflect the cleaned dataset
        dataset.cleaned_file = os.path.relpath(cleaned_file_path, settings.MEDIA_ROOT)
        dataset.status = 'processed'
        dataset.save()

        return JsonResponse({
            'message': 'Data cleaned and saved successfully!',
            'rows_affected': len(df) - len(cleaned_df)
        })

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    except Exception as e:
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


import json
import pandas as pd
from django.shortcuts import render, get_object_or_404
from .models import Dataset
import pandas as pd
import json
from django.shortcuts import render, get_object_or_404
from .models import Dataset  # Adjust based on your actual model import

from django.shortcuts import render, get_object_or_404
import pandas as pd
import json
from django.shortcuts import render, get_object_or_404
from .models import Dataset

from django.shortcuts import render, get_object_or_404
import pandas as pd
import json
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

from django.shortcuts import render
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from .models import Dataset, MLModel, DataPreprocessingLog
from django.db.models.functions import TruncMonth

import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_simple_distribution_plot(clean_count, unclean_count):
    # Ensure we're working with integers, defaulting to 0 for NaN values
    clean_count = int(clean_count) if pd.notna(clean_count) else 0
    unclean_count = int(unclean_count) if pd.notna(unclean_count) else 0
    
    # Only create plot if there's data to show
    if clean_count == 0 and unclean_count == 0:
        # Create an empty plot with a message
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No datasets available', horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    else:
        # Create figure and axis
        plt.figure(figsize=(8, 6))
        
        # Data
        labels = ['Clean', 'Unclean']
        sizes = [clean_count, unclean_count]
        colors = ['#22c55e', '#ef4444']  # Green and Red
        
        # Create pie chart
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Dataset Distribution', pad=20)
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Encode the bytes as base64
    return base64.b64encode(image_png).decode('utf-8')


def create_category_distribution_plot(datasets):
    plt.figure(figsize=(8, 4))
    
    # Define categories based on your data
    categories = {
        'Pending': datasets.filter(status='pending').count(),
        'Processed': datasets.filter(status='processed').count(),
        'With Target': datasets.exclude(target_class__isnull=True).count(),
        'Without Target': datasets.filter(target_class__isnull=True).count(),
        'Has Graphs': datasets.exclude(graphs__exact=[]).count(),
    }
    
    # Colors for each category
    colors = ['#3b82f6', '#10b981', '#6366f1', '#f59e0b', '#ef4444']
    
    # Create bar chart
    plt.bar(categories.keys(), categories.values(), color=colors)
    plt.title('Dataset Categories')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Datasets')
    
    # Add value labels on top of each bar
    for i, v in enumerate(categories.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')




def general_dashboard(request):
    if not request.user.is_authenticated:
        return redirect('signin')
        
    # Get all datasets for the current user
    datasets = Dataset.objects.filter(user=request.user)
    
    # Basic Statistics
    total_datasets = datasets.count()
    clean_datasets = datasets.filter(status='processed').count()
    unclean_datasets = total_datasets - clean_datasets
    total_models = MLModel.objects.count()
    
    # Calculate percentages
    clean_datasets_percentage = (clean_datasets / total_datasets * 100) if total_datasets > 0 else 0
    unclean_datasets_percentage = (unclean_datasets / total_datasets * 100) if total_datasets > 0 else 0
    
    # Calculate usage percentage for each dataset
    for dataset in datasets:
        # You can customize this based on your needs
        # For example, count how many times the dataset has been used in models
        usage_count = MLModel.objects.filter(dataset=dataset).count()
        max_usage = MLModel.objects.count() or 1  # Avoid division by zero
        dataset.usage_percentage = (usage_count / max_usage) * 100
    
    # Get current and previous month for comparison
    current_date = timezone.now()
    current_month_start = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    previous_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    
    # Dataset categories analysis (current month)
    current_month_stats = {
        'Diagnostic Imaging': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='diagnostic imaging'
        ).count(),
        'Data.Gov': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='data.gov'
        ).count(),
        'Image Net': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='image net'
        ).count(),
        'MNIST': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='mnist'
        ).count(),
        'MHLDDS': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='mhldds'
        ).count(),
        'HES': datasets.filter(
            uploaded_at__gte=current_month_start,
            description__icontains='hes'
        ).count(),
    }
    
    # Previous month stats
    previous_month_stats = {
        'Diagnostic Imaging': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='diagnostic imaging'
        ).count(),
        'Data.Gov': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='data.gov'
        ).count(),
        'Image Net': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='image net'
        ).count(),
        'MNIST': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='mnist'
        ).count(),
        'MHLDDS': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='mhldds'
        ).count(),
        'HES': datasets.filter(
            uploaded_at__gte=previous_month_start,
            uploaded_at__lt=current_month_start,
            description__icontains='hes'
        ).count(),
    }
    
    # Quality and Completeness Analysis
    # Get last 6 months of data
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
        total = stat['total']
        clean = stat['clean']
        completeness = (clean / total * 100) if total > 0 else 0
        # Assuming quality is based on some metrics in columns_info
        quality = completeness * 0.8  # Simplified quality metric
        completeness_data.append(completeness)
        quality_data.append(quality)
    
    # Dataset Usage Statistics
    datasets_usage = []
    for dataset in datasets:
        usage_count = MLModel.objects.filter(dataset=dataset).count()
        if usage_count > 0:
            datasets_usage.append({
                'title': dataset.name,
                'usage_percentage': min((usage_count / total_datasets * 100), 100),
                'count': usage_count
            })
    
    # Sort datasets_usage by count in descending order
    datasets_usage = sorted(datasets_usage, key=lambda x: x['count'], reverse=True)[:10]
    
    # Calculate average completeness and quality
    avg_completeness = sum(completeness_data) / len(completeness_data) if completeness_data else 0
    avg_quality = sum(quality_data) / len(quality_data) if quality_data else 0
    
    context = {
        'datasets': datasets,  # Changed from 'dataset' to 'datasets' to match template
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
        'datasets_usage': datasets_usage
    }
    
    return render(request, "general_dashboard.html", context)

### MODEL TRAINING PART ###

import json
import pandas as pd
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import matplotlib.pyplot as plt
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
import json

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
        dataset_id = request.POST.get('datasetId')
        target_column = request.POST.get('targetColumn')
        model_type = request.POST.get('model', 'neural_network')
        train_split = float(request.POST.get('trainTestSplit', 80)) / 100
        
        # Get the dataset
        try:
            dataset = Dataset.objects.get(id=dataset_id)
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
                
                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                # Get feature importance for top terms
                feature_importance = pd.Series(
                    model.feature_log_prob_[1] - model.feature_log_prob_[0],
                    index=vectorizer.get_feature_names_out()
                ).sort_values(ascending=False)
                
                top_features = feature_importance.head(10).to_dict()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': top_features,
                    'num_features': X_train_tfidf.shape[1],
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }
            else:
                # For numerical data, use Gaussian NB
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }
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
                
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=1000,
                    random_state=42,
                    solver=request.POST.get('solver', 'lbfgs')
                )
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Generate scatter plot of actual vs predicted values
                plt.figure(figsize=(10, 8))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Actual vs Predicted Values')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'model_type': 'regression',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

        return JsonResponse({
            'success': True,
            'results': results
        })

    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        return JsonResponse({
            'error': f'An error occurred during training: {str(e)}'
        }, status=500)

# Model Training View
def train_model(request):
    if request.method == 'POST':
        try:
            # Fetch form data
            dataset_id = request.POST.get('datasetId')
            target_column = request.POST.get('targetColumn')
            model_name = request.POST.get('model')
            train_test_split_ratio = int(request.POST.get('trainTestSplit', 80)) / 100

            # Fetch dataset
            dataset = get_object_or_404(Dataset, id=dataset_id)

            # Check if the dataset has been normalized
            # normalization_log = DataPreprocessingLog.objects.filter(
            #     dataset=dataset,
            #     action="Data Normalized"
            # ).exists()

            # if not normalization_log:
            #     return JsonResponse({'error': 'Dataset not Preprocessed. Please preprocess the dataset first.'}, status=400)
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

            # # Preprocess categorical features
            # categorical_columns = X.select_dtypes(include=['object']).columns
            # if not categorical_columns.empty:
            #     X = pd.get_dummies(X, columns=categorical_columns)

            # # Encode target column if it is categorical
            # if y.dtype == 'object':
            #     le = LabelEncoder()
            #     y = le.fit_transform(y)

            scaler = StandardScaler()
            # X_scaled = scaler.fit_transform(X)

            # Convert to NumPy arrays
            X = np.array(X)
            y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_split_ratio, random_state=42)

            # Initialize model variable
            model = None

            # Handle model selection and parameters
            # Handle model selection and parameters
            if model_name == 'linear_regression':
                # Fetch parameters for Linear Regression
                fit_intercept = request.POST.get('fitIntercept', 'true').lower() == 'true'
                model = LinearRegression(fit_intercept=fit_intercept)

                # Apply log transformation to the target
                # print("Applying log transformation to the target variable...")
                # y_train_log = np.log1p(y_train)  # log1p handles y_train=0 safely
                # y_test_log = np.log1p(y_test)

                # Scale the features
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit the model on the log-transformed target
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # # Inverse log transformation of predictions
                # y_pred = np.expm1(y_pred_log)  # expm1 reverses log1p

                # Evaluate metrics on the original scale
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results = {
                    'Mean Squared Error': mse,
                    'Mean Absolute Error': mae,
                    'R² Score': r2
                }
                print(f"Metrics: {results}")

                # Visualization
                plt.figure(figsize=(10, 6))
                plt.plot(y_test[:50], label='True', linestyle='-', marker='o')  # True values
                plt.plot(y_pred[:50], label='Predicted', linestyle='--', marker='x')  # Predicted values
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results with  Linear Regression")
                plt.xlabel("Sample")
                plt.ylabel("Value")
                
                # Save the visualization
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)
                plt.close()  # Close the plot to prevent overlapping in future plots

            elif model_name == 'decision_tree':
                # Fetch parameters for Decision Tree
                max_depth = request.POST.get('maxDepth', None)
                model = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # metrics = {'Model Accuracy': accuracy}

                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)

            elif model_name == 'svm':
                # Fetch parameters for SVM
                kernel = request.POST.get('kernel', 'rbf')
                model = SVC(kernel=kernel)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                try:
                    # Attempt to calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    # metrics = {'Model Accuracy': accuracy}

                    # Generate confusion matrix visualization
                    plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    
                    # Save plot to base64 string
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                    
                    results = {
                        'accuracy': float(accuracy),
                        'model_type': 'classification',
                        'feature_importance': None,
                        'num_features': len(feature_columns),
                        'features_used': feature_columns,
                        'visualization': f'data:image/png;base64,{image_base64}'
                    }
                except Exception as e:
                    # If an exception occurs, calculate alternative metrics
                    print(f"Accuracy score could not be calculated: {str(e)}")
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results = {
                        'Error': f"Accuracy score could not be calculated: {str(e)}",
                        'Mean Squared Error': mse,
                        'R² Score': r2
                    }
                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)

            elif model_name == 'random_forest':
                # Fetch parameters for Random Forest
                n_estimators = int(request.POST.get('nEstimators', 100))
                model = RandomForestClassifier(n_estimators=n_estimators)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # metrics = {'Model Accuracy': accuracy}

                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)
            elif model_name == 'knn':
                # Fetch parameters for k-Nearest Neighbors
                n_neighbors = int(request.POST.get('nNeighbors', 5))
                print('neighbors:', n_neighbors)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # metrics = {'Model Accuracy': accuracy}

                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)

            elif model_name == 'polynomial_regression':
                print("Polynomial Regression model selected.")

                # Fetch degree from request and log it
                degree = int(request.POST.get('degree', 2))
                print(f"Degree of polynomial: {degree}")

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Initialize PolynomialFeatures and transform training data
                poly = PolynomialFeatures(degree=degree)
                print("Transforming X_train into polynomial features...")
                X_train_poly = poly.fit_transform(X_train)
                print(f"Shape of X_train after transformation: {X_train_poly.shape}")

                # Transform test data and log its shape
                print("Transforming X_test into polynomial features...")
                X_test_poly = poly.transform(X_test)
                print(f"Shape of X_test after transformation: {X_test_poly.shape}")

                # Train the Linear Regression model
                print("Initializing and training LinearRegression model...")
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                print("Model training complete.")

                # Make predictions and log predictions shape
                print("Making predictions on X_test_poly...")
                y_pred = model.predict(X_test_poly)
                print(f"Shape of y_pred: {y_pred.shape}")
                print(f"First 5 predictions: {y_pred[:5]}")

                # Calculate metrics
                print("Calculating metrics...")
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results = {'Mean Squared Error': mse, 'R² Score': r2}
                print(f"Mean Squared Error: {mse}")
                print(f"R² Score: {r2}")

                # Plot results
                print("Plotting results...")
                plt.plot(y_test[:50], label='True')
                plt.plot(y_pred[:50], label='Predicted')  # Use X_test_poly here
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                
                # Save visualization
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)
                print(f"Visualization saved at {visualization_path}")

                model_path = os.path.join('media/models', f"{model_name}_{dataset_id}.joblib")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(model, model_path)

                return JsonResponse({
                        'success': True,
                        'results': results,
                })


            elif model_name == 'logistic_regression':
                solver = request.POST.get('solver', 'lbfgs')
                max_iter = int(request.POST.get('maxIter', 1000))
                model = LogisticRegression(solver=solver, max_iter=max_iter)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # metrics = {'Model Accuracy': accuracy}

                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)

            elif model_name == 'naive_bayes':
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # metrics = {'Model Accuracy': accuracy}

                # Generate confusion matrix visualization
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                results = {
                    'accuracy': float(accuracy),
                    'model_type': 'classification',
                    'feature_importance': None,
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
                plt.legend()
                plt.title(f"{model_name.capitalize()} Results")
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)

            elif model_name == 'kmeans':
                # Fetch the number of clusters
                n_clusters = max(2, int(request.POST.get('nClusters', 3)))  # Ensure n_clusters >= 2

                # Ensure feature scaling
                # scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train-test split on scaled features
                X_train, X_test = train_test_split(X_scaled, test_size=1 - train_test_split_ratio, random_state=42)

                # Initialize and train the KMeans model
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X_train)  # Fit on training data only

                # Predict clusters for both training and test data
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate clustering
                try:
                    silhouette_train = silhouette_score(X_train, y_train_pred)
                    silhouette_test = silhouette_score(X_test, y_test_pred)
                except ValueError as e:
                    return JsonResponse({'error': f"Silhouette score calculation failed: {str(e)}"}, status=400)

                results = {
                    'Silhouette Score (Train)': silhouette_train,
                    'Silhouette Score (Test)': silhouette_test,
                    'Inertia': model.inertia_,
                }

                # Visualization: Reduce to 2D using PCA
                pca = PCA(n_components=2)
                X_test_pca = pca.fit_transform(X_test)
                cluster_centers_pca = pca.transform(model.cluster_centers_)

                # Plot results
                plt.figure(figsize=(8, 6))
                plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pred, cmap='viridis', s=50, alpha=0.7, label='Data Points')
                plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
                plt.legend()
                plt.title(f"KMeans Clustering Results (n_clusters={n_clusters})")

                print(f"X_test shape: {X_test.shape}, X_test_pca shape: {X_test_pca.shape}")
                print(f"Cluster centers (before PCA): {model.cluster_centers_}")
                print(f"Cluster centers (after PCA): {cluster_centers_pca}")
                print(f"Unique cluster labels: {np.unique(y_test_pred)}")


                # Save visualization
                visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                plt.savefig(visualization_path)


            else:
                return JsonResponse({'error': 'Invalid model selected.'}, status=400)

            # Train model
            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            # accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            model_path = os.path.join('media/models', f"{model_name}_{dataset_id}.joblib")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)

            # Generate metrics
            score = model.score(X_test, y_test)
            # metrics = {'Model Accuracy': accuracy}
            # print(f"Model Accuracy: {metrics}")
            # print(f"Accuracy: {accuracy:.4f}")

            # Visualization
            # plt.figure(figsize=(8, 6))
            # if model_name == 'kmeans':
            #     # plt.scatter(X_test[:, 0], X_test[:, 1], c=model.predict(X_test), cmap='viridis')
            #     print("model is being executed")
            # else:
        #     plt.plot(y_test[:50], label='True')  # Fixed
        #     plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
            # plt.legend()
            # plt.title(f"{model_name.capitalize()} Results")
            # visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
            # os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
            # plt.savefig(visualization_path)

            return JsonResponse({
                'success': True,
                'results': results
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)



def training_page(request):
    datasets = Dataset.objects.all()

    return render(request, 'model_training.html', {'dataset': datasets})

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