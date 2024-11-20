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
        Dataset.objects.create(
            name=dataset_meta['name'],
            description=dataset_meta['description'],
            file_path=file_path,
            columns_info=columns,
            status="uploaded"
        )

        # Return a success response
        return JsonResponse({"success": True, "message": "Dataset created successfully!"})

    # Render the template with column details
    return render(request, 'datasets/create_dataset_step2.html', {
        'columns': columns,
        'row_range': range(num_rows),
        'dataset_meta': dataset_meta,
    })
from io import StringIO
from django.core.files.base import ContentFile
import uuid
import os
import openpyxl
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage


def create_dataset_interactive(request):
   
    return render(request, 'datasets/create_dataset_interactive.html')




def my_view(request):
    return render(request, 'dashboard.html')

def upload_file(request):
    if request.method == 'POST':
        source = request.POST.get('source')
        name = request.POST.get('name')
        description = request.POST.get('description')

        if source == 'local':
            dataset = request.FILES.get('dataset')
            if not dataset:
                return JsonResponse({'error': 'No file provided.'})
            
            # Save the dataset locally
            file_path = default_storage.save(f"datasets/{dataset.name}", dataset)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # Preview data
            try:
                if dataset.name.endswith('.csv'):
                    df = pd.read_csv(full_path)
                elif dataset.name.endswith(('.xls', '.xlsx')):
                    if not openpyxl:
                        return JsonResponse({'error': "Missing optional dependency 'openpyxl'. Please install it using 'pip install openpyxl'."})
                    df = pd.read_excel(full_path)
                else:
                    return JsonResponse({'error': 'Unsupported file format.'})
                preview = df.head().to_dict()
                return JsonResponse({'message': 'File uploaded successfully.', 'data_preview': preview})
            except Exception as e:
                return JsonResponse({'error': f"Error processing file: {str(e)}"})

        elif source == 'kaggle':
            kaggle_link = request.POST.get('kaggle_link')
            if not kaggle_link:
                return JsonResponse({'error': 'No Kaggle link provided.'})

            # Example: download dataset using Kaggle API
            # Ensure the Kaggle API is configured on the server
            try:
                dataset_name = kaggle_link.split('/')[-1]  # Extract dataset name
                os.system(f'kaggle datasets download -d {dataset_name} -p {settings.MEDIA_ROOT}/datasets')
                return JsonResponse({'message': 'Kaggle dataset downloaded successfully.'})
            except Exception as e:
                return JsonResponse({'error': f"Error downloading from Kaggle: {str(e)}"})

        else:
            return JsonResponse({'error': 'Invalid source selection.'})

    return render(request, 'upload.html')
