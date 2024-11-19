from django.shortcuts import render
import csv
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Dataset
from io import StringIO
from django.core.files.base import ContentFile
import uuid

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
