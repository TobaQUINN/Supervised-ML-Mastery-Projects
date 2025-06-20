# This script is for modeling the California housing dataset after pre-processing and feature engineering. The (Pre-Modeling.py) script has already prepared the dataset, and this script will focus on splitting the data, training various regression models, and evaluating their performance.

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your cleaned data
df = pd.read_csv('california_housing_dataset_final.csv')
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing sets to CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index= False)
y_train.to_csv('y_train.csv', index= False)
y_test.to_csv('y_test.csv', index= False)

# Display the shapes of the training and testing sets
print("Training sets shapes:", X_train.shape)
print("Test sets shapes:", X_test.shape)
print("Training target shapes:", y_train.shape)
print("Test target shapes:", y_test.shape)


# TRAINING MODELS
## a. Linear Regression
from sklearn.linear_model import LinearRegression
#Initialize the model
model_lr = LinearRegression()
# Fit the model to the training data
model_lr.fit(X_train, y_train)
# Predict on the test set
y_pred_lr = model_lr.predict(X_test)
#Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_score_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression -MSE: {mse_lr:.2f}, R^2: {r2_score_lr:.2f}')


# Scaling the features
## The features are already partially scaled, but it'sa good practice to standardize them for ridge, lasso, and other regression models.
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Now, we can proceed with Ridge and Lasso regression using the scaled features.
## b. Ridge Regression
from sklearn.linear_model import Ridge
#Initialize the model with a regularization parameter
model_ridge = Ridge(alpha=1.0)
# Fit the model to training data
model_ridge.fit(X_train_scaled, y_train)
#Predict on the test set
y_pred_ridge = model_ridge.predict(X_test_scaled)
#Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_score_ridge = r2_score(y_test, y_pred_ridge)
print(f'\nRidge Regression - MSE: {mse_ridge:.2f}, R^2: {r2_score_ridge:.2f}')


## c. Lasso Regression
from sklearn.linear_model import Lasso
#Initialize the model with a regularization parameter
model_lasso = Lasso(alpha=0.1)
# Fit the model to training data
model_lasso.fit(X_train_scaled, y_train)
#Predict on the test set
y_pred_lasso = model_lasso.predict(X_test_scaled)
#Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(f'\nLasso Regression - MSE: {mse_lasso:.2f}, R^2: {r2_score_lasso:.2f}')


#Visualizing the predictions of the models
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for seaborn
sns.set(style='darkgrid')
# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Linear Regression': y_pred_lr,
    'Ridge Regression': y_pred_ridge,
    'Lasso Regression': y_pred_lasso
})
# Melt DataFrame
predictions_melted = predictions_df.melt(var_name='Model', value_name='Predicted', id_vars='Actual')
# Create a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=predictions_melted, x='Actual', y='Predicted', hue='Model', alpha=0.7)
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend(title='Model')
plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.savefig('Model_Predictions_vs_Actual.png', bbox_inches='tight', dpi=300)
plt.show()

# Summary of the Model Performance
print("\nModel Performance Summary:")
print(f"Linear Regression - MSE: {mse_lr:.2f}, R^2: {r2_score_lr:.2f}")
print(f"Ridge Regression - MSE: {mse_ridge:.2f}, R^2: {r2_score_ridge:.2f}")
print(f"Lasso Regression - MSE: {mse_lasso:.2f}, R^2: {r2_score_lasso:.2f}")

# Save the model performance metrics to a CSV file
model_performance = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'MSE': [mse_lr, mse_ridge, mse_lasso],
    'R^2': [r2_score_lr, r2_score_ridge, r2_score_lasso]
})
model_performance.to_csv('model_performance.csv', index=False)

# Save the trained models using joblib
import os
import joblib
# Create a directory to save the models
os.makedirs('models', exist_ok=True)

joblib.dump(model_lr, 'models/linear_regression_model.pkl')
joblib.dump(model_ridge, 'models/ridge_regression_model.pkl')
joblib.dump(model_lasso, 'models/lasso_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')



# This Concludes the modeling process for the California housing dataset. The models have been trained, evaluated, and saved for future use. The performance metrics have been summarized and saved to a CSV file for further analysis or reporting.
