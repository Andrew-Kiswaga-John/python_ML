from django import forms

DATA_TYPES = [
    ('int', 'Integer'),
    ('float', 'Float'),
    ('string', 'String'),
    ('date', 'Date'),
    ('time', 'Time'),
]

from django import forms

class DatasetMetaForm(forms.Form):
    name = forms.CharField(
        max_length=255, 
        label="Dataset Name", 
        widget=forms.TextInput(attrs={
            'class': 'mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500'
        })
    )
    
    description = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500'
        }), 
        label="Description"
    )
    
    num_columns = forms.IntegerField(
        label="Number of Columns", 
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500'
        })
    )
    
    num_rows = forms.IntegerField(
        label="Number of Rows", 
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500'
        })
    )
