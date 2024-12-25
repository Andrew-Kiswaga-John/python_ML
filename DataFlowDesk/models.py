from django.db import models
from django.contrib.auth.models import User

# Profiles Table
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, null=True)
    # profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    website = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.user.username

# Datasets Table
class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_path = models.FileField(upload_to='datasets/')
    columns_info = models.JSONField(blank=True, null=True)  # JSON format for column details
    cleaned_file = models.FileField(upload_to='datasets/cleaned/', null=True, blank=True)
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('processed', 'Processed')], default='pending')
    graphs = models.JSONField(default=list, blank=True)  # New column to store chart file paths
    target_class = models.CharField(max_length=255, null=True, blank=True)  # Store the target column name
    dataset_type = models.CharField(max_length=255, null=True, blank=True)  # Store the target column name

    def __str__(self):
        return self.name

# Data Preprocessing Log Table
class DataPreprocessingLog(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='preprocessing_logs')
    action = models.CharField(max_length=255)  # e.g., "Missing Value Imputation"
    parameters = models.JSONField(blank=True, null=True)  # Details about preprocessing step
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.action} on {self.dataset.name}"

# ML Models Table
class MLModel(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='ml_models')
    algorithm = models.CharField(max_length=255)  # e.g., "Linear Regression"
    model_path = models.FileField(upload_to='models/', max_length=10000)  # Store hyperparameter details
    training_status = models.CharField(max_length=50, choices=[('training', 'Training'), ('completed', 'Completed')], default='training')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} on {self.dataset.name}"

# Model Results Table
class ModelResult(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='results')
    metric_name = models.CharField(max_length=255)  # e.g., "Accuracy", "RMSE"
    metric_value = models.FloatField()
    visualization_path = models.FileField(upload_to='visualizations/', blank=True, null=True)
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.metric_name}: {self.metric_value}"

# Exported Files Table
class ExportedFile(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='exported_files')
    file_type = models.CharField(max_length=255)  # e.g., "Trained Model", "Dataset"
    file_path = models.FileField(upload_to='exports/')
    exported_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_type} - {self.model.algorithm}"

# Tutorials Table
class Tutorial(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title