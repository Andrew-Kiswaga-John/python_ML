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

@ensure_csrf_cookie
def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({'success': True})
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
            
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

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
                file_path=file_path,
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
from django.http import JsonResponse
import plotly.express as px

from django.http import JsonResponse
import pandas as pd
from .models import Dataset

from django.http import JsonResponse
import plotly.express as px
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



from io import StringIO
from django.core.files.base import ContentFile
import uuid
import os
import pylightxl
import pandas as pd
from django.shortcuts import render
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

@csrf_exempt  # Optional, depending on your CSRF configuration
def upload_file(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        try:
            # Handling local file uploads
            if request.POST.get('source') == 'local':
                file = request.FILES.get('dataset')
                if not file:
                    return JsonResponse({'error': 'No file uploaded.'})

                # Ensure it's a CSV or Excel file
                if not (file.name.endswith('.csv') or file.name.endswith(('.xls', '.xlsx'))):
                    return JsonResponse({'error': 'Invalid file type. Please upload a CSV or Excel file.'})

                # Save the file to the datasets folder
                datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
                os.makedirs(datasets_dir, exist_ok=True)
                file_path = os.path.join(datasets_dir, file.name)

                with open(file_path, 'wb') as f:
                    for chunk in file.chunks():
                        f.write(chunk)

                # Load the file using pandas
                if file.name.endswith('.csv'):                   
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip')  # Default UTF-8
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='ISO-8859-1')  # Fallback to Latin-1
                        except Exception as e:
                            return JsonResponse({'error': f'File reading error: {str(e)}'})

                    # Handle cases where the delimiter isn't a comma
                    if df.empty or len(df.columns) == 1:  # Single-column issue
                        try:
                            df = pd.read_csv(file_path, delimiter=';')  # Try a semicolon
                        except Exception as e:
                            return JsonResponse({'error': f'CSV Parsing Error with fallback delimiter: {str(e)}'})
                        
                elif file.name.endswith('.xls'):
                        try:                           
                            df = pd.read_excel(file_path, engine='xlrd')
                        except Exception as e:
                            return JsonResponse({'error': f'File reading error: {str(e)}'})
                else:
                    # read excel
                    db = pylightxl.readxl(file_path)

                    # data sheet
                    name_first_sheet = db.ws_names[0]
                    sheet_data = list(db.ws(ws=name_first_sheet).rows)

                    # init dataframe
                    df = pd.DataFrame(data=sheet_data[1:], columns=sheet_data[0])

                # Save dataset details to the database
                dataset_instance = Dataset.objects.create(
                    user=request.user,
                    name=name,
                    description=description,
                    file_path=file_path,  # Path relative to MEDIA_ROOT
                    columns_info=df.columns.tolist(),  # Save column names as JSON
                )

                # Get the first five rows as HTML
                first_five_rows = df.head().to_html(classes='table table-bordered')
                print(f"dataset_id: {dataset_instance.id}")

                return JsonResponse({
                    'message': 'Dataset uploaded successfully!',
                    'preview': first_five_rows,
                    'dataset_id': dataset_instance.id
                })                            

            # Handling Kaggle dataset URL
            elif request.POST.get('source') == 'kaggle':
                kaggle_link = request.POST.get('kaggle_link')
                if not kaggle_link:
                    return JsonResponse({'error': 'No Kaggle link provided.'})

                # Extract dataset identifier from Kaggle link
                dataset_id = '/'.join(kaggle_link.rstrip('/').split('/')[-2:])
                datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
                os.makedirs(datasets_dir, exist_ok=True)

                # Use Kaggle API to download the dataset
                try:
                    import kaggle
                    kaggle.api.dataset_download_files(dataset_id, path=datasets_dir, unzip=True)
                except Exception as e:
                    return JsonResponse({'error': f'Error downloading dataset from Kaggle: {str(e)}'})

                # Locate the downloaded file
                downloaded_files = [
                    os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir)
                    if f.endswith('.csv') or f.endswith(('.xls', '.xlsx'))
                ]
                if not downloaded_files:
                    return JsonResponse({'error': 'No valid CSV or Excel files found in the downloaded dataset.'})

                file_path = downloaded_files[0]  # Use the first valid file

                # Load the dataset
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)

                # Save dataset details to the database
                dataset_instance = Dataset.objects.create(
                    user=request.user,
                    name=name,
                    description=description,
                    file_path=file_path,  # Path relative to MEDIA_ROOT
                    columns_info=df.columns.tolist(),  # Save column names as JSON
                )

                # Get the first five rows as HTML
                first_five_rows = df.head().to_html(classes='table table-bordered')
                print(f"dataset_id: {dataset_instance.id}")

                return JsonResponse({
                    'message': 'Dataset uploaded successfully!',
                    'preview': first_five_rows,
                    'dataset_id': dataset_instance.id
                })

            else:
                return JsonResponse({'error': 'Invalid data source selected.'})
        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'})

    return JsonResponse({'error': 'Invalid request method.'})


def clean_dataset(df, delete_header=False):
    """
    Cleans the dataset:
    - Handles duplicates, missing values, outliers, encoding, normalization
    - Includes preprocessing for boolean columns
    """
    print("Starting dataset cleaning...")

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



from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from django.utils.timezone import now
from django.utils.timezone import now
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from django.utils.timezone import now

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
from collections import defaultdict

# import matplotlib.pyplot as plt

from django.shortcuts import render, get_object_or_404
import pandas as pd
import json
import numpy as np
from collections import Counter

def generate_colors(n):
    """Generates a list of n visually distinct colors."""
    import numpy as np
    import matplotlib.colors as mcolors
    colors = list(mcolors.CSS4_COLORS.values())
    if n > len(colors):  # Repeat colors if necessary
        colors *= (n // len(colors)) + 1
    np.random.shuffle(colors)
    return colors[:n]


def analyze_column_relationships(data):
    """Analyze relationships between columns to recommend appropriate graph types."""
    relationships = {}
    
    # Patterns for different column types
    text_patterns = ['message', 'description', 'comment', 'text', 'note']
    id_patterns = ['id', 'code', 'uuid', 'ref']
    category_patterns = ['category', 'type', 'class', 'label', 'status', 'group', 'spam', 'survived']
    numeric_patterns = ['age', 'price', 'amount', 'count', 'number', 'score', 'rate']
    
    for col in data.columns:
        column_data = data[col].dropna()
        unique_count = len(column_data.unique())
        total_rows = len(column_data)
        
        relationships[col] = {
            'type': 'unknown',
            'unique_count': unique_count,
            'compatible_graphs': [],
            'recommended_as_x': False,
            'recommended_as_y': False,
            'is_text_heavy': False,
            'is_identifier': False,
            'total_rows': total_rows,
            'unique_ratio': unique_count / total_rows if total_rows > 0 else 0,
            'distribution_friendly': False,  # New flag for columns good for distribution analysis
            'can_be_counted': False  # New flag to indicate if column can be used for counting
        }
        
        # Skip empty columns
        if total_rows == 0:
            continue
        
        # Check if column is an identifier
        is_identifier = any(pattern in col.lower() for pattern in id_patterns)
        relationships[col]['is_identifier'] = is_identifier
        
        # Simple text detection
        if data[col].dtype == object:
            # Check if it's a text-heavy column
            is_text_heavy = (
                any(pattern in col.lower() for pattern in text_patterns) or
                any(len(str(x)) > 50 for x in column_data.head(5))
            )
            relationships[col]['is_text_heavy'] = is_text_heavy
            
            if is_text_heavy:
                relationships[col]['type'] = 'text'
                continue
            
            # Enhanced categorical detection
            is_categorical = (
                any(pattern in col.lower() for pattern in category_patterns) or
                unique_count <= 20 or
                relationships[col]['unique_ratio'] <= 0.2
            )
            
            if is_categorical:
                relationships[col]['type'] = 'categorical'
                relationships[col]['compatible_graphs'] = ['bar', 'distribution']
                relationships[col]['recommended_as_x'] = True
                relationships[col]['distribution_friendly'] = True
                relationships[col]['can_be_counted'] = True  # Categorical columns can be counted
                if unique_count <= 10:
                    relationships[col]['compatible_graphs'].append('pie')
            else:
                relationships[col]['type'] = 'text'
        
        # Handle numeric data
        elif pd.api.types.is_numeric_dtype(column_data):
            # Check if it's a numeric column good for distribution analysis
            is_distribution_friendly = any(pattern in col.lower() for pattern in numeric_patterns)
            relationships[col]['distribution_friendly'] = is_distribution_friendly
            
            # If it's an ID or has unique values equal to row count, mark as identifier
            if is_identifier or unique_count == total_rows:
                relationships[col]['type'] = 'identifier'
            else:
                # Check if it's a binary/categorical numeric column
                if unique_count <= 2:
                    relationships[col]['type'] = 'categorical'
                    relationships[col]['compatible_graphs'] = ['pie', 'bar', 'distribution']
                    relationships[col]['recommended_as_x'] = True
                    relationships[col]['distribution_friendly'] = True
                    relationships[col]['can_be_counted'] = True  # Binary columns can be counted
                elif unique_count <= 20:
                    relationships[col]['type'] = 'discrete'
                    relationships[col]['compatible_graphs'] = ['bar', 'line', 'distribution']
                    relationships[col]['recommended_as_x'] = True
                    relationships[col]['recommended_as_y'] = True
                    relationships[col]['distribution_friendly'] = True
                    relationships[col]['can_be_counted'] = True  # Discrete columns can be counted
                else:
                    relationships[col]['type'] = 'continuous'
                    relationships[col]['compatible_graphs'] = ['scatter', 'line', 'distribution']
                    relationships[col]['recommended_as_y'] = True
                    relationships[col]['distribution_friendly'] = True
        
        # Handle datetime data
        elif pd.api.types.is_datetime64_any_dtype(column_data):
            relationships[col]['type'] = 'datetime'
            relationships[col]['compatible_graphs'] = ['line', 'scatter']
            relationships[col]['recommended_as_x'] = True
    
    return relationships

def get_recommended_graphs(x_col, y_col, relationships):
    """Get recommended graph types based on column relationships."""
    x_info = relationships[x_col]
    recommendations = []
    
    # For categorical x-axis with no y-axis (distribution analysis)
    if x_info['type'] == 'categorical' and not y_col:
        recommendations.append({
            'type': 'bar',
            'confidence': 0.95,
            'reason': f'Shows count distribution of {x_col}',
            'requires_y': False
        })
        if x_info['unique_count'] <= 10:
            recommendations.append({
                'type': 'pie',
                'confidence': 0.9,
                'reason': 'Best for showing proportion of categories',
                'requires_y': False
            })
    
    # For categorical x-axis with numerical y-axis
    elif x_info['type'] == 'categorical' and y_col:
        y_info = relationships[y_col]
        if y_info['type'] in ['continuous', 'discrete']:
            recommendations.append({
                'type': 'distribution',
                'confidence': 0.95,
                'reason': f'Best for analyzing {y_col} distribution across {x_col} categories',
                'requires_y': True
            })
            recommendations.append({
                'type': 'box',
                'confidence': 0.9,
                'reason': f'Good for comparing {y_col} statistics across {x_col} categories',
                'requires_y': True
            })
            recommendations.append({
                'type': 'bar',
                'confidence': 0.85,
                'reason': f'Shows average {y_col} for each {x_col} category',
                'requires_y': True
            })
    
    # For numerical or datetime x-axis with numerical y-axis
    elif x_info['type'] in ['continuous', 'discrete', 'datetime'] and y_col:
        y_info = relationships[y_col]
        if y_info['type'] in ['continuous', 'discrete']:
            if x_info['type'] == 'datetime':
                recommendations.append({
                    'type': 'line',
                    'confidence': 0.95,
                    'reason': 'Best for showing trends over time',
                    'requires_y': True
                })
            recommendations.append({
                'type': 'scatter',
                'confidence': 0.9,
                'reason': 'Best for showing relationships between numerical variables',
                'requires_y': True
            })
            if x_info['type'] == 'discrete' or y_info['type'] == 'discrete':
                recommendations.append({
                    'type': 'bar',
                    'confidence': 0.85,
                    'reason': 'Good for comparing discrete values',
                    'requires_y': True
                })
    
    return recommendations

def display_graphs(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    try:
        # Load and preprocess data
        if dataset.status == 'processed':
            data = pd.read_csv(dataset.cleaned_file)
        else:
            data = pd.read_csv(dataset.file_path)
        
        # Basic data cleaning
        data.columns = [str(col).strip() for col in data.columns]
        data = data.dropna(axis=1, how='all')
        
        # Try to convert date strings to datetime
        for col in data.columns:
            if data[col].dtype == object:
                try:
                    data[col] = pd.to_datetime(data[col], errors='raise')
                except (ValueError, TypeError):
                    continue
        
        # Analyze columns
        column_relationships = analyze_column_relationships(data)
        
        # Find default columns for initial display
        default_x = None
        default_y = None
        
        # First, try to find a datetime column for x-axis
        for col, info in column_relationships.items():
            if info['type'] == 'datetime':
                default_x = col
                break
        
        # If no datetime, try to find a categorical column
        if not default_x:
            for col, info in column_relationships.items():
                if (info['type'] == 'categorical' and 
                    not info['is_text_heavy'] and 
                    not info['is_identifier']):
                    default_x = col
                    break
        
        # Find a numerical column for y-axis
        for col, info in column_relationships.items():
            if (info['recommended_as_y'] and 
                not info['is_identifier'] and 
                col != default_x):
                default_y = col
                break
        
        # Prepare data for template
        sample_size = min(100, len(data))
        dataset_data = data.head(sample_size).values.tolist()
        columns = data.columns.tolist()
        
        context = {
            'dataset': dataset,
            'dataset_data': json.dumps(dataset_data),
            'columns': json.dumps(columns),
            'column_relationships': json.dumps(column_relationships),
            'default_x': json.dumps(default_x),
            'default_y': json.dumps(default_y)
        }
        
        return render(request, 'graphs.html', context)
    
    except Exception as e:
        return JsonResponse({
            'error': f'Error processing dataset: {str(e)}'
        }, status=500)


# Function to render the upload.html page
def upload_page(request):
    return render(request, 'upload.html')



def all_datasets(request) :
    dataset = Dataset.objects.all()

    return render(request, "datasets/show_datasets.html", {'dataset' : dataset})


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
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
                )
            else:
                # For numerical data, use Gaussian NB
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
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
                
                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=1000,
                    random_state=42,
                    solver=request.POST.get('solver', 'lbfgs')
                )
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='regression',
                    y_true=y_test,
                    y_pred=y_pred
                )

        # Save the trained model as a .pkl file
        model_filename = f"Neural_network_model_{dataset_id}.pkl"
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
            X = np.array(X)
            y = np.array(y)

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
                y_pred = model.predict(X_test)

                # Evaluate metrics on the original scale
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='regression',
                    y_true=y_test,
                    y_pred=y_pred
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


            elif model_name == 'decision_tree':
                # Fetch parameters for Decision Tree
                max_depth = request.POST.get('maxDepth', 3)
                model = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Generate visualizations
                feature_names = feature_columns  # Feature names from the dataset
                class_names = df[target_column].unique().astype(str)  # Class names based on unique target labels
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred,
                    model=model,
                    feature_names=feature_names,
                    class_names=class_names
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



            elif model_name == 'svm':
                # Fetch parameters for SVM
                kernel = request.POST.get('kernel', 'rbf')
                model = SVC(kernel=kernel)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                try:
                    # Generate Visualizations
                    results = generate_visualizations(
                        model_type='classification',
                        y_true=y_test,
                        y_pred=y_pred
                    )
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

                # Save the trained model as a .pkl file
                model_filename = f"svm_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='SVM',
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
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
                )

                # Save the trained model as a .pkl file
                model_filename = f"random_forest_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Random Forest',
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
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
                )

                # Save the trained model as a .pkl file
                model_filename = f"knn_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='KNN',
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
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='regression',
                    y_true=y_test,
                    y_pred=y_pred
                )

                # Save the trained model as a .pkl file
                model_filename = f"Polynomial_regression_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Polynomial Regression',
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
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
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


            elif model_name == 'naive_bayes':
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Generate Visualizations
                results = generate_visualizations(
                    model_type='classification',
                    y_true=y_test,
                    y_pred=y_pred
                )

                # Save the trained model as a .pkl file
                model_filename = f"naive_bayes_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='Naive Bayes',
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

            elif model_name == 'kmeans':
                # Fetch the number of clusters
                n_clusters = max(2, int(request.POST.get('nClusters', 3)))  # Ensure n_clusters >= 2
                X = df
                # Ensure feature scaling
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

                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                results = {
                    'Silhouette Score (Train)': silhouette_train,
                    'Silhouette Score (Test)': silhouette_test,
                    'Inertia': model.inertia_,
                    'model_type': 'clustering',
                    'num_features': len(feature_columns),
                    'features_used': feature_columns,
                    'visualization': f'data:image/png;base64,{image_base64}'
                }

                # Save the trained model as a .pkl file
                model_filename = f"kmeans_model_{dataset_id}.pkl"
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    joblib.dump(model, f)

                # Save to MLModel table
                ml_model = MLModel.objects.create(
                    dataset=dataset,
                    algorithm='KMeans',
                    training_status='completed',
                    model_path=model_path
                )

                # Save results to ModelResult table
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='Silhouette Score (Train)',
                    metric_value=silhouette_train,
                )
                ModelResult.objects.create(
                    model=ml_model,
                    metric_name='Silhouette Score (Test)',
                    metric_value=silhouette_test,
                )



            else:
                return JsonResponse({'error': 'Invalid model selected.'}, status=400)

            return JsonResponse({
                'success': True,
                'results': results,
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def generate_visualizations(model_type, y_true=None, y_pred=None, model=None, feature_names=None, class_names=None, **kwargs):

    # buffer = io.BytesIO()
    visualizations = {}
    additional_metrics = {}

    if model_type == 'classification':
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        buffer = io.BytesIO()
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        visualizations['confusion_matrix'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        plt.close()

        # Generate decision tree visualization (if applicable)
        if model and isinstance(model, DecisionTreeClassifier):
            buffer = io.BytesIO()
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
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            visualizations['decision_tree'] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
            plt.close()

        # Accuracy metric
        accuracy = accuracy_score(y_true, y_pred)
        additional_metrics['accuracy'] = round(accuracy, 3)

    elif model_type == 'regression':
        # Scatter Plot: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, label='Predicted vs Actual')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Ideal Fit')
        plt.title('Regression Results: Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        visualizations['linear_graph'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # Metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        additional_metrics = {
            'mean_squared_error': round(mse, 3),
            'r2_score': round(r2, 3)
        }

    elif model_type == 'clustering':
        # Cluster Visualization using PCA (if provided in kwargs)
        if 'pca_data' in kwargs:
            pca_data = kwargs['pca_data']
            clusters = kwargs.get('clusters', None)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                hue=clusters,
                palette='viridis',
                alpha=0.7
            )
            plt.title('Cluster Visualization')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend(title="Cluster")
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            visualizations['clusters_visual'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

        # Additional Clustering Metrics
        additional_metrics = {
            'inertia': kwargs.get('inertia', None),
            'silhouette_score': kwargs.get('silhouette_score', None)
        }

    return {
        'visualizations': visualizations,
        'metrics': additional_metrics,
        'model_type': model_type
    }


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

def render_predictions_view(request):
    # Fetch models to display in the dropdown
    models = MLModel.objects.filter(training_status='completed').order_by('-created_at')
    return render(request, 'predictions.html', {
        'models': models,
        'target_columns': None,  # Initial state; no dataset uploaded
        'results': None,         # No predictions yet
        'metrics': None,         # No metrics yet
    })


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



# def predictions_view(request):
#     models = MLModel.objects.filter(training_status='completed').order_by('-created_at')
#     target_columns = None
#     results_html = None
#     metrics = None

#     if request.method == 'POST' and request.FILES.get('dataset'):
#         try:
#             # Handle dataset upload
#             uploaded_file = request.FILES['dataset']
#             file_extension = uploaded_file.name.split('.')[-1].lower()

#             # Read the dataset based on the file type
#             if file_extension == 'csv':
#                 df = pd.read_csv(uploaded_file, on_bad_lines='skip')
#             elif file_extension in ['xls', 'xlsx']:
#                 df = pd.read_excel(uploaded_file)
#             else:
#                 return JsonResponse({'error': 'Unsupported file format. Please upload a CSV or Excel file.'})
#             dataset = clean_dataset(df, delete_header=False)

#             # Extract column names for target column selection
#             target_columns = dataset.columns.tolist()

#             # Handle predictions
#             model_id = request.POST['model']
#             model_entry = models.get(id=model_id)
#             model_path = model_entry.model_path.path
#             model_algorithm = model_entry.algorithm

#             if not os.path.exists(model_path):
#                 return JsonResponse({'error': 'Model file not found.'})

#             # Load the trained model
#             model = joblib.load(model_path)

#             # Prepare features
#             X = dataset
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)

#             # Check if the model is KMeans
#             if model_algorithm == 'KMeans':
#                 # Predict cluster labels
#                 cluster_labels = model.predict(X_scaled)
                
#                 # Append cluster labels to dataset
#                 dataset['Cluster'] = cluster_labels

#                 # Evaluate clustering if silhouette score is applicable
#                 try:
#                     silhouette = silhouette_score(X_scaled, cluster_labels)
#                     metrics = {
#                         "type": "clustering",
#                         "silhouette_score": round(silhouette, 3),
#                         "inertia": round(model.inertia_, 3),
#                     }
#                 except ValueError as e:
#                     metrics = {
#                         "type": "clustering",
#                         "error": f"Silhouette score calculation failed: {str(e)}",
#                         "inertia": round(model.inertia_, 3),
#                     }

#                 # (Optional) PCA for visualization
#                 pca = PCA(n_components=2)
#                 X_pca = pca.fit_transform(X_scaled)
#                 dataset['PCA_1'] = X_pca[:, 0]
#                 dataset['PCA_2'] = X_pca[:, 1]

#             else:
#                 # Handle other models (Polynomial Regression, etc.)
#                 target_column = request.POST['target']
#                 X = dataset.drop(columns=[target_column])
#                 y = dataset[target_column]

#                 if model_algorithm == 'Polynomial Regression':
#                     poly = PolynomialFeatures(degree=2)
#                     X_transformed = poly.fit_transform(X_scaled)
#                 else:
#                     X_transformed = X_scaled

#                 predictions = model.predict(X_transformed)

#                 # Determine if it's regression or classification
#                 if hasattr(model, "predict_proba") or len(set(y)) <= 2:
#                     accuracy = accuracy_score(y, predictions)
#                     cm = confusion_matrix(y, predictions)
#                     metrics = {
#                         "type": "classification",
#                         "accuracy": round(accuracy, 3),
#                         "confusion_matrix": cm.tolist(),
#                     }
#                 else:
#                     mse = mean_squared_error(y, predictions)
#                     r2 = r2_score(y, predictions)
#                     metrics = {
#                         "type": "regression",
#                         "mse": round(mse, 3),
#                         "r2": round(r2, 3),
#                     }

#                 # Append predictions to dataset
#                 dataset['Predicted'] = predictions

#             # Convert dataset to HTML for rendering
#             results_html = dataset.to_html(classes='table-auto w-full text-center border-collapse')

#         except Exception as e:
#             return JsonResponse({'error': str(e)})

#     return render(request, 'predictions.html', {
#         'models': models,
#         'target_columns': target_columns,
#         'results': results_html,
#         'metrics': metrics,
#     })

