# -*- coding: utf-8 -*-
"""
Spam Email Classifier

This script trains a machine learning model to classify emails as spam or ham
using Logistic Regression and TF-IDF vectorization.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "mail_data.csv"  # Ensure this file is in the same directory
mail_data = pd.read_csv(data_path)

# Handle missing values by replacing NaN with empty strings
mail_data.fillna("", inplace=True)

# Display dataset shape
# print(f"Dataset Shape: {mail_data.shape}")

# Encode labels: 'spam' -> 0, 'ham' -> 1
mail_data["Category"] = mail_data["Category"].map({"spam": 0, "ham": 1})

# Split features and labels
X = mail_data["Message"]
y = mail_data["Category"].astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature extraction using TF-IDF vectorization
vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, y_train)

# Evaluate model accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train_features))
test_accuracy = accuracy_score(y_test, model.predict(X_test_features))

# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Testing Accuracy: {test_accuracy:.4f}")

# Predict spam or ham for a new email
def classify_email(email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return "Spam Mail" if prediction[0] == 0 else "Not Spam Mail"

# Example email classification
sample_email = "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"
result = classify_email(sample_email)
print(f"{result}")

