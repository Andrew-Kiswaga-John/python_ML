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

def display_dataset(request, id):
    # Fetch the dataset from the database
    dataset = Dataset.objects.get(id=id)
    data = pd.read_csv(dataset.file_path)  # Load the dataset from CSV

    # Basic statistics of the dataset
    stats = data.describe().to_dict()

    # Create a graph using Plotly (example: scatter plot of two columns)
    fig = px.scatter(data_frame=data, x=data.columns[0], y=data.columns[1], title="Dataset Scatter Plot")
    graph_html = fig.to_html(full_html=False)

    # Convert the dataset rows to a list of lists
    dataset_data = data.values.tolist()

    return render(request, 'datasets/display_dataset.html', {
        'dataset': dataset,
        'dataset_data': dataset_data,
        'table_html': data.to_html(classes='table table-striped', index=False),
        'stats': stats,
        'graph_html': graph_html,  # Pass the graph HTML to the template
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
# Function to render the upload.html page
def upload_page(request):
    return render(request, 'upload.html')