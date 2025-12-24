# Telecustomer Churn Prediction

A machine learning project to predict customer churn for a telecom provider. The project includes data ingestion, preprocessing, model selection with hyperparameter tuning, and artifact export for downstream prediction.

Note: The main project lives in the subfolder `Telecustomer-churn-prediction-main/`. This root README provides quick start instructions and links.

## Quick Start

- Prerequisites
  - Python 3.9+ recommended
  - Windows PowerShell or any terminal

- Setup (Windows PowerShell)
  - Navigate to the repo root:
    - `cd c:\Users\vkr10\Downloads\Telecustomer-churn-prediction-main`
  - Create and activate a virtual environment (optional but recommended):
    - `python -m venv .venv`
    - `./.venv/Scripts/Activate.ps1`
  - Install dependencies:
    - `pip install -r Telecustomer-churn-prediction-main/requirements.txt`

- Prepare data
  - Place your dataset CSV at:
    - `Telecustomer-churn-prediction-main/notebook/data/Customerchurn.csv`
  - The ingestion script reads this file and creates train/test splits under `Telecustomer-churn-prediction-main/artifacts/`.

- Train model (end-to-end)
  - From repo root:
    - `python Telecustomer-churn-prediction-main/src/components/data_ingestion.py`
  - What this does:
    - Ingests data and splits into train/test (`artifacts/train.csv`, `artifacts/test.csv`)
    - Transforms features (one-hot encoding, scaling, feature selection)
    - Trains multiple models with hyperparameter search and SMOTE oversampling
    - Saves the best model to `Telecustomer-churn-prediction-main/artifacts/model.pkl`
    - Saves the preprocessing pipeline to `Telecustomer-churn-prediction-main/artifacts/preprocessor.pkl`

## Predict Usage (Python example)

```python
import pandas as pd
from src.utils import load_object

# Load artifacts (paths relative to inner project folder)
preprocessor = load_object('Telecustomer-churn-prediction-main/artifacts/preprocessor.pkl')
model = load_object('Telecustomer-churn-prediction-main/artifacts/model.pkl')

# Prepare a DataFrame with the same schema as train data
sample = pd.DataFrame([
    {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 50.0,
        'TotalCharges': 600.0,
    }
])

X = preprocessor.transform(sample)
pred = model.predict(X)
print('Churn prediction:', 'Yes' if pred[0] == 1 else 'No')
```

## Project Structure

The core code and assets reside in `Telecustomer-churn-prediction-main/`:

- `artifacts/` — intermediate files & trained artifacts (`model.pkl`, `preprocessor.pkl`)
- `notebook/data/` — raw dataset location (`Customerchurn.csv`)
- `src/components/` — pipeline components
  - `data_ingestion.py` — reads raw data and creates train/test splits
  - `data_transformation.py` — preprocessing and feature engineering
  - `model_trainer.py` — model training, evaluation, and saving best model
- `src/utils.py` — helpers for saving/loading objects, evaluating models
- `requirements.txt` — dependencies

## Notes

- Ensure the dataset column names match the expected schema used in `data_transformation.py`.
- If using HTTPS to push code, you may need a GitHub Personal Access Token when prompted for password.
- For reproducibility, prefer running inside a virtual environment.

## Links

- Detailed README: `Telecustomer-churn-prediction-main/README.md`
- Repository: https://github.com/vasanthkumarreddy1/Telecustomer-churn-prediction-main