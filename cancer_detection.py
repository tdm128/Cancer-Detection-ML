import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading the Excel dataset
cancer_data_set = pd.read_excel('Cancer_Data.xlsx')

# Check if the dataset is loaded correctly
if cancer_data_set.empty:
    raise ValueError("The dataset is empty. Please check the file path or the contents of 'Cancer_Data.xlsx'.")

# Show the first few rows to verify data
print("First few rows of the dataset:")
print(cancer_data_set.head())

# Convert 'diagnosis' to 0 for Malignant (M) and 1 for Benign (B)
cancer_data_set['diagnosis'] = cancer_data_set['diagnosis'].replace({'M': 0, 'B': 1})

# Handle missing values - instead of dropping them, let's fill with column means
# This is more robust than dropping rows, which might be removing too many samples
cancer_data_set.fillna(cancer_data_set.mean(), inplace=True)

# Check the shape of the dataset after handling missing values
print("Shape of dataset after filling missing values:", cancer_data_set.shape)

# ID is not needed for prediction, so drop it
if 'id' in cancer_data_set.columns:
    cancer_data_set.drop(columns=['id'], inplace=True)

# Define features (X) and target variable (Y)
X = cancer_data_set.drop(columns=['diagnosis'])
Y = cancer_data_set['diagnosis']

# Ensure there are no NaN values left
if X.isnull().values.any() or Y.isnull().values.any():
    raise ValueError("There are still NaN values present in the data after handling missing values.")

# Check if there are any samples left after data processing
if X.shape[0] == 0 or Y.shape[0] == 0:
    raise ValueError("No samples left for training after processing. Please check the dataset or processing steps.")

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Ensure the train-test split has worked properly
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the training data
Y_train_predictions = model.predict(X_train)

# Calculate the accuracy on the training data
training_data_accuracy = accuracy_score(Y_train, Y_train_predictions)
print('Accuracy on training data:', training_data_accuracy)

# Make predictions on the test data
Y_test_predictions = model.predict(X_test)

# Calculate the accuracy on the test data
test_data_accuracy = accuracy_score(Y_test, Y_test_predictions)
print('Accuracy on test data:', test_data_accuracy)

# Prediction on unseen data (example input)
input_data = (13.7, 17.64, 87.76, 571.1, 0.0995, 0.07957, 0.04548, 0.0316, 0.1732, 0.06088,
              0.2431, 0.9462, 1.564, 20.64, 0.003245, 0.008186, 0.01698, 0.009233, 0.01285,
              0.001524, 14.96, 23.53, 95.78, 686.5, 0.1199, 0.1346, 0.1742, 0.09077, 0.2518, 0.0696)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array for a single prediction (1, -1)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = model.predict(input_data_reshaped)

# Interpret prediction result
if prediction[0] == 0:
    print('The cancer is Malignant')
else:
    print('The cancer is Benign')