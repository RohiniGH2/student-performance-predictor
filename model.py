import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import xgboost as xgb

# Loading the dataset
df = pd.read_csv('dataset.csv')

# One-hot encoding for the categorical features
cat_cols = ['Gender', 'Socioeconomic status', 'Participation in extracurricular activities']
X = df.drop(['Final grade (out of 100)'], axis=1)
y = df[['Final grade (out of 100)']]

X_encoded = pd.get_dummies(X, columns=cat_cols)

# We split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Hyperparameter grid for XGBoost model
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [5, 7, 10]
}


# Training this XGBoost model for final grade
xgbm = xgb.XGBRegressor(random_state=42, tree_method='hist', verbosity=0)
grid = GridSearchCV(xgbm, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())
best_model = grid.best_estimator_


# We save the model
joblib.dump(best_model, 'final_grade_xgb_model.pkl')

# Prediction function
def predict_student_performance(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, columns=cat_cols)
    input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)
    return best_model.predict(input_df)[0]