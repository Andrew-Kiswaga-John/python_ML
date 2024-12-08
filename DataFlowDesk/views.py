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
import pandas as pd
import json
from django.shortcuts import render, get_object_or_404
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

def display_dataset(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    # Load dataset file
    if dataset.status == 'processed':
        data = pd.read_csv(dataset.cleaned_file)
    else:
        # data = pd.read_csv(dataset.file_path)
        try:
            data = pd.read_csv(dataset.file_path, on_bad_lines='skip')  # Default UTF-8
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(dataset.file_path, on_bad_lines='skip', encoding='ISO-8859-1')  # Fallback to Latin-1
            except Exception as e:
                return JsonResponse({'error': f'File reading error: {str(e)}'})

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
            
                

            # Handling Kaggle dataset URL (future implementation)
            elif request.POST.get('source') == 'kaggle':
                return JsonResponse({'error': 'Kaggle source is not implemented yet.'})

            else:
                return JsonResponse({'error': 'Invalid data source selected.'})
        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'})

    return JsonResponse({'error': 'Invalid request method.'})


def clean_dataset(df, delete_header=False):
    """
    Cleans the dataset:
    - Optionally deletes the header and uses first data row as new header
    - Removes duplicate rows
    - Fills missing numerical values with the column mean
    - Fills missing categorical values with 'Unknown'
    - Ensures proper data types for numerical and categorical columns
    """
    print("Starting dataset cleaning...")

    # If user wants to delete header
    if delete_header:
        print("Deleting header and using first data row as new header...")
        # Store current column names
        original_columns = df.columns.tolist()
        # Get the first data row to use as new header
        new_headers = df.iloc[0].values.tolist()
        # Drop the first row and reset index
        df = df.iloc[1:].reset_index(drop=True)
        # Set the new headers
        df.columns = new_headers
        print("Header replaced with first data row.")
    
    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - df.shape[0]
    print(f"Removed {removed_duplicates} duplicate rows.")

    # Handle missing values
    for col in df.columns:
        # Debug: Log column name and type
        print(f"Processing column: {col} (Type: {df[col].dtype})")

        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in numerical column '{col}' with mean.")
                df[col].fillna(df[col].mean(), inplace=True)

        # Handle object/categorical columns
        elif pd.api.types.is_object_dtype(df[col]):
            num_missing = df[col].isnull().sum()
            if num_missing > 0:
                print(f"Filling {num_missing} missing values in categorical column '{col}' with 'Unknown'.")
                df[col].fillna('Unknown', inplace=True)

        # Handle columns that don't match numeric or object types
        else:
            print(f"Skipping column '{col}' as it does not fit numeric or categorical types.")

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

        # Load dataset with original headers
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # Perform cleaning with the delete_header parameter
        cleaned_df = clean_dataset(df, delete_header=delete_header)

        # Save the cleaned dataset with a new name
        cleaned_datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'cleaned')
        os.makedirs(cleaned_datasets_dir, exist_ok=True)
        cleaned_file_name = os.path.basename(file_path).replace('.csv', '_cleaned.csv')
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

        file_path = dataset.file_path  # Path to the dataset

        # Load dataset
        df = pd.read_csv(file_path, on_bad_lines='skip')

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
from django.utils.timezone import now
from django.utils.timezone import now

from sklearn.preprocessing import MinMaxScaler
from django.utils.timezone import now

def perform_data_normalization(request, dataset_id):
    # Ensure only POST requests are allowed for normalization
    if request.method == 'POST':
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Access the actual file path using .path
        file_path = dataset.file_path.path
        
        # Load the dataset
        df = pd.read_csv(file_path)  # Adjust this line based on how your dataset is stored
        
        # Assuming you want to normalize all numerical columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Perform Min-Max normalization (scale values between 0 and 1)
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Save to a new file with a timestamp to avoid overwriting
        new_file_path = f"{file_path.replace('.csv', '')}_normalized_{now().strftime('%Y%m%d%H%M%S')}.csv"
        df.to_csv(new_file_path, index=False)  # Save with a new name
        
        # Optionally, you can update the dataset object with the new file path
        dataset.file_path = new_file_path
        dataset.save()
        
        # Optionally, return a message to the frontend
        return JsonResponse({'message': f'Data normalized successfully. Saved as {new_file_path}'})
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
from django.shortcuts import render, get_object_or_404
from .models import Dataset  # Adjust based on your actual model import

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



def all_datasets(request):
    dataset = Dataset.objects.filter(user=request.user)
    return render(request, "datasets/show_datasets.html", {'dataset': dataset})



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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import joblib
import os


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
def train_model(request):
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

def training_page(request):
    datasets = Dataset.objects.all()
    return render(request, 'model_training.html', {'dataset': datasets})

def get_columns(request):
    if request.method == 'GET':
        dataset_id = request.GET.get('datasetId')
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            if dataset.status == 'processed':
                dataset_path = dataset.cleaned_file
            else:            
                dataset_path = dataset.file_path

            data = pd.read_csv(dataset_path)
            columns = list(data.columns)
            return JsonResponse({'columns': columns})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)