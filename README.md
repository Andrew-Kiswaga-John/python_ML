# DataFlowDesk - Machine Learning Platform

DataFlowDesk is a comprehensive web-based machine learning platform built with Django that enables users to manage datasets, train ML models, and visualize results through an intuitive interface.

## Features

- **Dataset Management**
  - Upload and manage datasets
  - Automated data cleaning and preprocessing
  - Data visualization and analysis tools
  - Download cleaned datasets

- **Machine Learning**
  - Support for multiple ML algorithms
  - Neural Network implementation with PyTorch
  - Automated model training and evaluation
  - Model performance visualization
  - Model export functionality

- **Visualization**
  - Interactive data visualizations
  - Model performance metrics
  - Training progress tracking
  - Cluster analysis visualization

- **User Management**
  - User authentication and profiles
  - Personal dataset and model management
  - Tutorial system for platform guidance

## Technology Stack

- **Backend**: Django
- **Frontend**: HTML, TailwindCSS
- **Machine Learning**: 
  - PyTorch (Deep Learning)
  - Scikit-learn (Traditional ML)
  - Pandas (Data Processing)
  - Matplotlib/Seaborn (Visualization)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
4. Run migrations:
   ```bash
   python manage.py migrate
   ```
5. Start the development server:
   ```bash
   python manage.py runserver
   ```

## Project Structure

- `DataFlowDesk/` - Main Django application
  - `models.py` - Database models for datasets, ML models, and user profiles
  - `views.py` - Core logic for ML operations and web interface
  - `templates/` - HTML templates
  - `static/` - Static files (CSS, JS, images)
  - `templatetags/` - Custom template tags

## Features in Detail

### Data Processing
- Automated data cleaning
- Missing value handling
- Outlier detection
- Feature scaling and normalization
- Data visualization tools

### Machine Learning Models
- Classification algorithms
- Regression algorithms
- Clustering (K-means)
- Neural Networks
- Model evaluation metrics

### Visualization Tools
- Data distribution plots
- Correlation matrices
- Learning curves
- ROC curves
- Confusion matrices
- Cluster visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.