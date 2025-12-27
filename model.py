import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import xgboost as xgb

# Load dataset
df = pd.read_csv('dataset.csv')

# One-hot encode categorical features
cat_cols = ['Gender', 'Socioeconomic status', 'Participation in extracurricular activities']
X = df.drop(['Final grade (out of 100)'], axis=1)
y = df[['Final grade (out of 100)']]

X_encoded = pd.get_dummies(X, columns=cat_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Hyperparameter grid for XGBoost
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [5, 7, 10]
}


# Train XGBoost model for final grade
xgbm = xgb.XGBRegressor(random_state=42, tree_method='hist', verbosity=0)
grid = GridSearchCV(xgbm, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())
best_model = grid.best_estimator_

# Cross-validation R2
cv_r2 = cross_val_score(best_model, X_train, y_train.values.ravel(), cv=5, scoring='r2')
print(f"\nFinal grade - Best Params: {grid.best_params_}")
print(f"CV R2: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Test set evaluation
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test.values.ravel(), y_pred)
rmse = np.sqrt(mean_squared_error(y_test.values.ravel(), y_pred))
print(f"Test R2 Score = {r2:.4f}, Test RMSE = {rmse:.4f}")

# Save model
joblib.dump(best_model, 'final_grade_xgb_model.pkl')
print("\n Model saved for final grade.")

# Prediction function
def predict_student_performance(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, columns=cat_cols)
    input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)
    return best_model.predict(input_df)[0]