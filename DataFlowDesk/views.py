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
import pylightxl
import pandas as pd
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
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

@csrf_exempt  # Optional, depending on your CSRF configuration
def upload_file(request):
    if request.method == 'POST':
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

                # Get the first five rows as HTML
                first_five_rows = df.head().to_html(classes='table table-bordered')

                return JsonResponse({
                    'message': 'Dataset uploaded successfully!',
                    'preview': first_five_rows
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