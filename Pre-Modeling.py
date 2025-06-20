## Data Collection and Preparation
# This script demonstrates how I collected and prepared data for modeling using the california housing dataset.
# It includes loading the dataset, exploring it, and preparing it for modeling.

from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# View data
print(df.head())
print(df.describe())
print(df.info())

#Check for missing values
print("Number of missing values:", df.isnull().sum())
# Drop rows and columns with missing values
df.dropna(inplace=True)

# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())
# Drop duplicate rows
df.drop_duplicates(inplace=True)

## No need to convert categorical variables to numerical as there are no categorical variables in this dataset.
## No duplicate/missing columns found in the dataset.

#Check Data Types
print("Data types:\n", df.dtypes)

#Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Checking Feature Correlations
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()  # The plot shows strong positive correlation between 'MedInc' and 'MedHouseVal', 
            ## 'AveRooms' and 'AveBedrms' show multicollinearity, which may affect model performance, so I will drop one of them
plt.savefig('Housing_Price_Feature_Correlations.png', bbox_inches='tight', dpi=300)

# Fishing out Key features for modeling and dropping multicollinear features
##'MedInc' is a strong predictor of the target variable
###  Drop 'AveBedrms' due to multicollinearity with 'AveRooms'
df.drop(columns=['AveBedrms'], inplace=True)
df.rename(columns={'AveRooms': 'AvgRooms'}, inplace= True) # Renaming for Clarity
df.drop(columns=['AveOccup', 'Population'], inplace=True) # Dropping 'AveOccup' and 'Population' as they are less relevant

#Feature Engineering to combine features 'Latitude' and 'Longitude' into a single feature 'Location_Distance'
# Define Central point
central_lat, central_lon = df['Latitude'].mean(), df['Longitude'].mean()
# Caalculate distance from central point
df['Location_Distance'] = np.sqrt((df['Latitude'] - central_lat)**2 + (df['Longitude']- central_lon)**2)
# Drop Original Latitude and Longitude Columns
df.drop(columns= ['Latitude', 'Longitude'], inplace=True)
# Final Data Preparation
df.reset_index(drop=True, inplace=True) # Reset index after dropping rows/columns


# Save the cleaned and prepared dataset to a CSV file
df.to_csv('california_housing_dataset_cleaned.csv', index= False)



# Visualize the distribution of target variable
target = 'MedHouseVal' # This is the Target Variable Y Dependent Variable
features= df.drop(columns=[target]).columns.tolist() # All other columns are features X Independent Variables
df[target] = df[target].astype(float) # Ensure Target is float for correlation analysis
plt.figure(figsize=(10, 6))
sns.histplot(df[target], bins= 30, kde=True)
plt.title('Distibution of target variable')
plt.xlabel('Median House Value')
plt.ylabel('Frequnecy')
plt.savefig('Target_Variable_Distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# Save the feature and the target variables to seperate csv files
df[features].to_csv('California_Housing_Features.csv', index=False)
df[target].to_csv('California_Housing_Target.csv', index=False)
# Save the cleaned DataFrame to a CSV file
df.to_csv('california_housing_dataset_final.csv', index=False)

# Display the final DataFrame
print("Final DataFrame:\n", df.head())



# Summary of the Data Preparation
print("\nData Preparation Summary:")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print(f"Features: {features}")
print(f"Target Variable: {target}")



# End of Data Preparation Script
# The dataset is now ready for modeling with the following features:
#- MedInc: Median Income in block group
# - HouseAge: Median House Age in block group
# - AvgRooms: Average number of rooms per household
# - Population: Total population in block group
# - Location_Distance: Distance from the central point based on latitude and longitude
# - MedHouseVal: Median House Value in block group (target variable)
# The target variable is 'MedHouseVal', which is the median house value in the block group.


