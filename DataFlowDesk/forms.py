from django import forms

DATA_TYPES = [
    ('int', 'Integer'),
    ('float', 'Float'),
    ('string', 'String'),
    ('date', 'Date'),
    ('time', 'Time'),
]

class DatasetMetaForm(forms.Form):
    name = forms.CharField(max_length=255, label="Dataset Name")
    description = forms.CharField(widget=forms.Textarea, label="Description")
    num_columns = forms.IntegerField(label="Number of Columns")
    num_rows = forms.IntegerField(label="Number of Rows")