from django.shortcuts import render
from django.shortcuts import render, redirect
from .forms import DatasetMetaForm
from django.http import JsonResponse

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

def display_graphs(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    if dataset.status == 'processed':
        data = pd.read_csv(dataset.cleaned_file)
    else:
        data = pd.read_csv(dataset.file_path)

    # Determine column types
    column_types = {}
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].dtype == 'int64':
                column_types[column] = 'discrete'
            else:
                column_types[column] = 'continuous'
        else:
            column_types[column] = 'categorical'

    dataset_data = data.head(100).values.tolist()
    columns = data.columns.tolist()

    return render(request, 'graphs.html', {
        'dataset': dataset,
        'dataset_data': json.dumps(dataset_data),
        'columns': json.dumps(columns),
        'column_types': json.dumps(column_types),
    })



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

            # Check if the dataset is processed
            if dataset.status != 'processed':
                return JsonResponse({'error': 'Dataset not processed. Please clean the dataset first.'}, status=400)

            # Load dataset
            df = pd.read_csv(dataset.cleaned_file.path)

            # Validate target column
            if target_column not in df.columns:
                return JsonResponse({'error': f"Target column '{target_column}' does not exist in the dataset."}, status=400)

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Preprocess categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                X = pd.get_dummies(X, columns=categorical_columns)

            # Encode target column if it is categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Convert to NumPy arrays
            X = np.array(X_scaled)
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
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metrics = {'Mean Squared Error': mse, 'R² Score': r2}

            elif model_name == 'decision_tree':
                # Fetch parameters for Decision Tree
                max_depth = request.POST.get('maxDepth', None)
                model = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'svm':
                # Fetch parameters for SVM
                kernel = request.POST.get('kernel', 'rbf')
                model = SVC(kernel=kernel)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                try:
                    # Attempt to calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    metrics = {'Model Accuracy': accuracy}
                except Exception as e:
                    # If an exception occurs, calculate alternative metrics
                    print(f"Accuracy score could not be calculated: {str(e)}")
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {
                        'Error': f"Accuracy score could not be calculated: {str(e)}",
                        'Mean Squared Error': mse,
                        'R² Score': r2
                    }

            elif model_name == 'random_forest':
                # Fetch parameters for Random Forest
                n_estimators = int(request.POST.get('nEstimators', 100))
                model = RandomForestClassifier(n_estimators=n_estimators)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'knn':
                # Fetch parameters for k-Nearest Neighbors
                n_neighbors = int(request.POST.get('nNeighbors', 5))
                print('neighbors:', n_neighbors)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'polynomial_regression':
                degree = int(request.POST.get('degree', 2))
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'logistic_regression':
                solver = request.POST.get('solver', 'lbfgs')
                max_iter = int(request.POST.get('maxIter', 100))
                model = LogisticRegression(solver=solver, max_iter=max_iter)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'naive_bayes':
                var_smoothing = float(request.POST.get('varSmoothing', 1e-9))
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metrics = {'Model Accuracy': accuracy}

            elif model_name == 'kmeans':
                n_clusters = int(request.POST.get('nClusters', 3))
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X_train)
                y_pred = model.predict(X_test)
                
                try:
                    # Attempt to calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    metrics = {'Model Accuracy': accuracy}
                except Exception as e:
                    # If an exception occurs, calculate regression metrics instead
                    print(f"Accuracy score could not be calculated: {str(e)}")
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {
                        'Error': f"Accuracy score could not be calculated: {str(e)}",
                        'Mean Squared Error': mse,
                        'R² Score': r2
                    }

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
            plt.figure(figsize=(8, 6))
            if model_name == 'kmeans':
                plt.scatter(X_test[:, 0], X_test[:, 1], c=model.predict(X_test), cmap='viridis')
            else:
                plt.plot(y_test[:50], label='True')  # Fixed
                plt.plot(model.predict(X_test)[:50], label='Predicted')  # Fixed
            plt.legend()
            plt.title(f"{model_name.capitalize()} Results")
            visualization_path = os.path.join('media/visualizations', f"{model_name}_{dataset_id}.png")
            os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
            plt.savefig(visualization_path)

            return JsonResponse({
                'metrics': metrics,
                'visualization': visualization_path,
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