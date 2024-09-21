import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

# Load the heart disease dataset
data = pd.read_csv('/content/heart.csv')

# NOTE: This dataset contains many missing values.
missing_values = data.isnull()
num_missing_values = missing_values.sum()

# Check for missing values
print(num_missing_values)

# Print the dataset
print(data)

# Interpolate missing values
interpolated_data = data.interpolate()
print(interpolated_data)

# Split the data into features and target variable
X = interpolated_data.drop('output', axis=1)
y = interpolated_data['output']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Calculate performance metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
