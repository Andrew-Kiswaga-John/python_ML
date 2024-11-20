from django.shortcuts import render
import csv
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Dataset
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
    if request.method == 'POST':
        if 'columns' in request.POST and 'rows' in request.POST:
            try:
                columns = request.POST.getlist('columns[]')
                rows = int(request.POST.get('rows'))
                # Prepare an empty table
                table = [['' for _ in range(len(columns))] for _ in range(rows)]
                return JsonResponse({'columns': columns, 'table': table})
            except ValueError:
                return JsonResponse({'error': 'Invalid input'}, status=400)

        # Step 2: Handle final submission to save as CSV
        elif 'final_data' in request.POST:
            try:
                dataset_name = request.POST.get('name')
                description = request.POST.get('description')
                data = request.POST.getlist('final_data[]')
                columns = request.POST.getlist('columns[]')
                rows = len(data) // len(columns)

                # Convert data into CSV format
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                for i in range(rows):
                    writer.writerow(data[i * len(columns):(i + 1) * len(columns)])

                # Save CSV to file field
                csv_file = ContentFile(output.getvalue().encode('utf-8'))
                file_name = f'{uuid.uuid4()}.csv'
                dataset = Dataset(
                    user=request.user,
                    name=dataset_name,
                    description=description,
                )
                dataset.file_path.save(file_name, csv_file)
                dataset.save()

                return JsonResponse({'success': 'Dataset created successfully', 'dataset_id': dataset.id})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

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
