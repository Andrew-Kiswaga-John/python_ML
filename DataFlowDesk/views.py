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



def display_dataset(request, id):
    # Fetch the dataset from the database using the 'id' (default primary key)
    # dataset = Dataset.objects.get(id=id)
    # data = pd.read_csv(dataset.file_path)  # Load the dataset from CSV

    # Fetch the dataset from the database using the 'id' (default primary key)
    try:
        dataset = Dataset.objects.get(id=id)
        print(f"Dataset fetched successfully: {dataset.name}")
    except Dataset.DoesNotExist:
        print(f"Dataset with ID {id} does not exist.")
        return JsonResponse({'error': 'Dataset not found.'}, status=404)

    # Check the status of the dataset and fetch the appropriate file path
    if dataset.status == 'processed':
        # file_path = os.path.join(settings.MEDIA_ROOT, dataset.cleaned_file)
        data = pd.read_csv(dataset.cleaned_file)
        print(f"Using cleaned file for dataset: {dataset.cleaned_file}")
    else:
        # data = pd.read_csv(dataset.file_path)
        try:
            data = pd.read_csv(dataset.file_path, on_bad_lines='skip')  # Default UTF-8
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(dataset.file_path, on_bad_lines='skip', encoding='ISO-8859-1')  # Fallback to Latin-1
            except Exception as e:
                return JsonResponse({'error': f'File reading error: {str(e)}'})
        print(f"Using original file for dataset: {dataset.file_path}")

    # # Load the dataset
    # try:
    #     data = pd.read_csv(file_path)
    #     print(f"Dataset loaded successfully. Shape: {data.shape}")
    # except Exception as e:
    #     print(f"Error loading dataset: {str(e)}")
    #     return JsonResponse({'error': f"Error loading dataset: {str(e)}"}, status=500)

    # Get the first 5 rows for initial display
    data_head = data.head()

    # Basic statistics of the dataset
    stats = data.describe().to_dict()

    # Create a graph using Plotly with custom styling
    fig = px.scatter(data_frame=data, x=data.columns[0], y=data.columns[1], title="Dataset Scatter Plot")
    fig.update_layout(
        title="Dataset Scatter Plot",
        title_x=0.5,
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=data.columns[0],
            title_font=dict(size=14),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=data.columns[1],
            title_font=dict(size=14),
            showgrid=True,
            zeroline=False,
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="gray"
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
    )

    # Convert the dataset rows to a list of lists (for the first 5 rows)
    dataset_data = data_head.values.tolist()

    # Handle the "Show More" request via AJAX
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == "GET":
        # Get the number of rows already displayed (stored in the session or via URL)
        displayed_rows = int(request.GET.get('displayed_rows', 5))

        # Fetch the next 5 rows
        next_rows = data.iloc[displayed_rows:displayed_rows+5].values.tolist()

        # Return the next rows as JSON
        return JsonResponse({'next_rows': next_rows})

    # Render the template with dataset and graph data
    return render(request, 'datasets/display_dataset.html', {
        'dataset': dataset,
        'dataset_data': dataset_data,  # First 5 rows for display
        'table_html': data_head.to_html(classes='table table-striped', index=False),
        'stats': stats,
        'graph_html': fig.to_html(full_html=False),
        'columns': data.columns,
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

# Preview the tasks for data cleaning
def data_cleaning_preview(request, dataset_id):
    # Fetch dataset from the database (mock example below)
    dataset = Dataset.objects.get(id=dataset_id)
    # Check the status of the dataset
    if dataset.status == 'processed':
        print("Dataset is already processed. Skipping cleaning tasks.")
        return JsonResponse({'tasks': ["Dataset already processed"]})


    file_path = dataset.file_path

    try:
        # Load the dataset
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # Generate cleaning tasks
        tasks = []
        if df.isnull().values.any():
            tasks.append(f"Found {df.isnull().sum().sum()} missing values. These will be handled.")
        if df.duplicated().sum() > 0:
            tasks.append(f"Found {df.duplicated().sum()} duplicate rows. These will be removed.")
        tasks.append(f"Columns with incorrect data types will be converted.")
        tasks.append(f"Outliers in numerical columns will be treated.")

        return JsonResponse({'tasks': tasks})
    except Exception as e:
        return JsonResponse({'error': f"Error processing file: {str(e)}"}, status=500)

# Perform the actual cleaning
def clean_dataset(df):
    """
    Cleans the dataset:
    - Removes duplicate rows.
    - Fills missing numerical values with the column mean.
    - Fills missing categorical values with 'Unknown'.
    - Ensures proper data types for numerical and categorical columns.
    """
    print("Cleaning dataset...")
    
    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - df.shape[0]
    print(f"Removed {removed_duplicates} duplicate rows.")
    
    # Fill missing numerical values with mean
    num_missing_numerical = df.select_dtypes(include=['number']).isnull().sum().sum()
    print(f"Numerical missing values before filling: {num_missing_numerical}")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Fill missing categorical values with 'Unknown'
    num_missing_categorical = df.select_dtypes(include=['object']).isnull().sum().sum()
    print(f"Categorical missing values before filling: {num_missing_categorical}")
    for col in df.select_dtypes(include=['object']).columns:
        df.loc[:, col] = df[col].fillna('Unknown').astype(str)  # Explicit use of loc

    print("Dataset cleaning complete.")
    return df


def perform_data_cleaning(request, dataset_id):
    """
    Handles the cleaning of a dataset, including an option to remove the first row.
    """
    if request.method == 'POST':
        print(f"Received POST request for data cleaning. Dataset ID: {dataset_id}")

        # Fetch the dataset
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            print(f"Dataset fetched successfully: {dataset.name}")
        except Dataset.DoesNotExist:
            print(f"Dataset with ID {dataset_id} does not exist.")
            return JsonResponse({'error': f"Dataset with ID {dataset_id} does not exist."}, status=404)

        file_path = dataset.file_path # Get the actual file path
        print(f"File path of the dataset: {file_path}")

        try:
            # Parse the JSON body to get the user's choice
            # body = json.loads(request.body)
            # remove_first_row = body.get('remove_first_row', False)  # Default to False
            # print(f"Remove first row: {remove_first_row}")
            try:
                # Debug: Log the raw body content
                print(f"Raw request body: {request.body}")

                # Parse JSON from the request body
                body = json.loads(request.body)
                remove_first_row = body.get('remove_first_row', False)
                print(f"Remove first row: {remove_first_row}")

                # (Continue with the rest of the function...)
                
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {str(e)}")
                return JsonResponse({'error': 'Invalid JSON payload received.'}, status=400)

            # Load the dataset
            print("Attempting to load dataset...")
            df = pd.read_csv(file_path, on_bad_lines='skip')
            print(f"Dataset loaded successfully. Shape: {df.shape}")

            # Remove the first row if requested
            if remove_first_row:
                print("Making the first row the header of the dataset.")
                df.columns = df.iloc[0]  # Set the first row as column names
                df = df[1:]  # Drop the first row
                df.reset_index(drop=True, inplace=True)  # Reset the index
                print("First row has been made the header successfully.")

            # Perform cleaning
            print("Starting data cleaning process...")
            cleaned_df = clean_dataset(df)
            print(f"Data cleaning completed. Cleaned DataFrame shape: {cleaned_df.shape}")

            # Create the cleaned datasets folder if it doesn't exist
            cleaned_datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'cleaned')
            os.makedirs(cleaned_datasets_dir, exist_ok=True)

            # Generate the path for the cleaned dataset
            cleaned_file_name = os.path.basename(file_path).replace('.csv', '_cleaned.csv')
            cleaned_file_path = os.path.join(cleaned_datasets_dir, cleaned_file_name)
            print(f"Saving cleaned dataset to: {cleaned_file_path}")

            # Save the cleaned dataset
            cleaned_df.to_csv(cleaned_file_path, index=False)

            # Update the dataset record
            dataset.cleaned_file = os.path.relpath(cleaned_file_path, settings.MEDIA_ROOT)  # Path relative to MEDIA_ROOT
            dataset.status = 'processed'  # Update the status
            dataset.save()
            print("Cleaned dataset saved successfully and dataset record updated.")

            return JsonResponse({'message': 'Data cleaned and saved successfully!'})
        except Exception as e:
            print(f"Error during data cleaning: {str(e)}")
            return JsonResponse({'error': f"Error during cleaning: {str(e)}"}, status=500)

    print("Invalid request method. Only POST is allowed.")
    return JsonResponse({'error': 'Invalid request method.'}, status=400)




# Function to render the upload.html page
def upload_page(request):
    return render(request, 'upload.html')