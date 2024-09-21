import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
water_data = pd.read_csv("/content/waterQuality1.csv")

# Preprocess the data
water_data = water_data.replace('#NUM!', pd.NA).dropna()

# Encode the target variable
label_encoder = LabelEncoder()
water_data['is_safe'] = label_encoder.fit_transform(water_data['is_safe'])

# Define features and target
X = water_data.drop('is_safe', axis=1)
y = water_data['is_safe']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# New data for prediction
new_data = pd.DataFrame([[0.6, 24.58, 0.01, 0.71, 0.005, 3.14, 0.77, 1.45, 0.98, 0.35, 0.002, 0.167, 14.66, 1.84, 0.004, 23.43, 4.99, 0.08, 0.25, 0.08]], columns=X.columns)
prediction = model.predict(new_data)

# Get the predicted class
predicted_class = label_encoder.inverse_transform(prediction)

# Output the result
if predicted_class == 'yes':
    print("The water is good.")
else:
    print("The water is not good.")
