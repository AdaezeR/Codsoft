# Titanic Survival Prediction Project

## Overview
This project analyzes passenger survival patterns from the RMS Titanic disaster using machine learning. The implementation includes data preprocessing, multiple classification models, hyperparameter tuning, and prediction capabilities to determine which factors most influenced survival outcomes.

## Features
- **Data Analysis**: Examines survival rates by passenger class, gender, and age
- **Machine Learning Pipeline**:
  - Preprocessing for numeric and categorical features
  - Multiple classification algorithms (Random Forest, SVM, Logistic Regression etc.)
  - Hyperparameter tuning with GridSearchCV
- **Model Interpretation**:
  - Feature importance analysis
  - Prediction probabilities
  - Performance metrics (accuracy, precision, recall)

## Key Findings
- **Top Performing Model**: Random Forest (82.12% accuracy)
- **Most Important Features**: 
  1. Fare amount (25.0% importance)
  2. Passenger age (24.6% importance) 
  3. Gender (14.5% female / 14.4% male)
- **Survival Patterns**:
  - First-class passengers had 63% survival rate vs 24% in third-class
  - 74% of women survived vs 19% of men
  - Children had higher survival rates (50%) than adults (38%)

## Usage
1. Install requirements:
```bash
pip install pandas scikit-learn xgboost lightgbm matplotlib

Run the Jupyter notebook:
jupyter notebook Titanic_Survival_Analysis.ipynb

Make new predictions:
new_data = pd.DataFrame({
    'Pclass': [2],
    'Sex': ['female'],
    'Age': [28],
    'SibSp': [0],
    'Parch': [1],
    'Fare': [35.50],
    'Embarked': ['S']
})
prediction = model.predict(new_data)

File Structure
titanic-survival/
├── data/
│   └── Titanic.csv          # Original dataset
├── notebooks/
│   └── Titanic_Analysis.ipynb  # Main analysis notebook
├── models/
│   └── best_model.pkl       # Serialized trained model
└── README.md                # This file

Requirements
Python 3.8+
pandas
scikit-learn
matplotlib
xgboost (optional)
lightgbm (optional)

Acknowledgments
Dataset sourced from Kaggle Titanic Competition
