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
    return render(request, 'datasets/create_dataset_step1.html', {'form': form})

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
    dataset_meta = request.session.get('dataset_meta')
    if not dataset_meta:
        return redirect('create_dataset_step1')  # Redirect back if meta is missing

    columns = dataset_meta['columns']  # Retrieve column details
    num_rows = dataset_meta['num_rows']

    if request.method == "POST":
        data = request.POST.getlist('data')  # All user inputs

        # Validate the data based on column types
        error_messages = []
        validated_data = [[] for _ in range(len(columns))]  # Initialize empty lists for each column

        # Group data by column
        for i, column in enumerate(columns):
            column_data = data[i::len(columns)]  # Split data by columns
            for j, value in enumerate(column_data):
                if column['type'] == 'int':
                    try:
                        validated_value = int(value)
                    except ValueError:
                        error_messages.append(f"Row {j+1}, Column '{column['name']}': Value must be an integer.")
                elif column['type'] == 'float':
                    try:
                        validated_value = float(value)
                    except ValueError:
                        error_messages.append(f"Row {j+1}, Column '{column['name']}': Value must be a float.")
                elif column['type'] == 'string':
                    validated_value = str(value)
                elif column['type'] == 'date':
                    try:
                        validated_value = datetime.strptime(value, "%Y-%m-%d").date()
                    except ValueError:
                        error_messages.append(f"Row {j+1}, Column '{column['name']}': Value must be a date in YYYY-MM-DD format.")
                elif column['type'] == 'time':
                    try:
                        validated_value = datetime.strptime(value, "%H:%M:%S").time()
                    except ValueError:
                        error_messages.append(f"Row {j+1}, Column '{column['name']}': Value must be a time in HH:MM:SS format.")
                
                # Append validated value to the appropriate column's list
                validated_data[i].append(validated_value)

        # If there are validation errors, return them
        if error_messages:
            return JsonResponse({"errors": error_messages}, status=400)

        # Ensure the datasets directory exists
        datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')  # Use MEDIA_ROOT as base
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)

        # Full file path for the CSV
        file_path = os.path.join(datasets_dir, f"{dataset_meta['name']}.csv")

        # Write the validated data to CSV, including the 'id' column
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['ID'] + [column['name'] for column in columns]  # Add 'ID' to the header
            writer.writerow(header)

            # Write rows of data, including the automatically generated 'ID' column
            for i in range(num_rows):
                row_data = [i + 1]  # The ID column, starting from 1 and incrementing
                for col_data in validated_data:
                    row_data.append(col_data[i])  # Add the data from each column
                writer.writerow(row_data)

        # Save the dataset information to the database
        new_dataset = Dataset.objects.create(
            name=dataset_meta['name'],
            description=dataset_meta['description'],
            file_path=file_path,
            columns_info=columns,
            status="uploaded"
        )

        # Redirect to the "Display Dataset" page
        return redirect(reverse('display_dataset', args=[new_dataset.id]))

    # Render the template with column details
    return render(request, 'datasets/create_dataset_step2.html', {
        'columns': columns,
        'row_range': range(num_rows),
        'dataset_meta': dataset_meta,
    })


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
        data = pd.read_csv(dataset.file_path)

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
    return render(request, 'dashboard.html')

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
                        df = pd.read_csv(file_path, on_bad_lines='skip')  # Skip problematic lines
                    except pd.errors.ParserError as e:
                        return JsonResponse({'error': f'CSV Parsing Error: {str(e)}'})

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


def clean_dataset(df):
    """
    Cleans the dataset:
    - Removes duplicate rows.
    - Fills missing numerical values with the column mean.
    - Fills missing categorical values with 'Unknown'.
    - Ensures proper data types for numerical and categorical columns.
    """
    print("Starting dataset cleaning...")

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
        # Fetch the dataset
        dataset = Dataset.objects.get(id=dataset_id)

        file_path = dataset.file_path.path  # Get the file path

        # Load dataset
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # Perform cleaning
        cleaned_df = clean_dataset(df)

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

        return JsonResponse({'message': 'Data cleaned and saved successfully!'})

    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f"Error during cleaning: {str(e)}"}, status=500)




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

def create_graphs(request, dataset_id):
    # Fetch the dataset
    dataset = get_object_or_404(Dataset, id=dataset_id)
    file_path = dataset.file_path.path
    df = pd.read_csv(file_path)

    charts = []

    # Line chart for numeric columns (using histograms or bar charts for numerical columns)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        for col in numeric_columns:
            # Generate a histogram for numerical data
            data_values = df[col].dropna()
            if len(data_values) > 10:  # Avoid charts for too small datasets
                bin_count = min(10, len(data_values) // 5)  # Dynamically adjust bin count
                # Create dynamic bins based on data range
                bins = pd.cut(data_values, bins=bin_count)
                bin_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in bins.cat.categories]
                
                # Group by bins and count occurrences
                binned_data = data_values.groupby(bins).size().tolist()

                chart_data = {
                    'type': 'bar',  # Could also use 'histogram' or 'line' based on dataset
                    'data': json.dumps({
                        'labels': bin_labels,  # Label bins dynamically
                        'datasets': [{
                            'data': binned_data,
                            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                            'borderColor': 'rgba(54, 162, 235, 1)',
                            'borderWidth': 1,
                        }],
                    }),
                    'title': f'Numeric Distribution: {col}',
                }
                charts.append(chart_data)

    # Pie charts for categorical columns (excluding columns with too many unique values or message columns)
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        unique_count = df[col].nunique()
        # Skip columns with too many unique values (e.g., a message column with unique text per row)
        if unique_count > 1 and unique_count < 30:  # Set a threshold for unique categories
            value_counts = df[col].value_counts()
            charts.append({
                'type': 'pie',
                'data': json.dumps({
                    'labels': value_counts.index.tolist(),
                    'datasets': [{
                        'data': value_counts.values.tolist(),
                        'backgroundColor': ['#FF6384', '#36A2EB', '#FFCE56'] * (len(value_counts) // 3 + 1),
                    }]
                }),
                'title': f'Category Distribution: {col}',
            })

    # Render the template with the charts
    return render(request, 'graphs.html', {'charts': charts})


# Function to render the upload.html page
def upload_page(request):
    return render(request, 'upload.html')



def all_datasets(request) :
    dataset = Dataset.objects.all()

    return render(request, "datasets/show_datasets.html", {'dataset' : dataset})

