# SPAM-MAIL-DETECTION
Spam Email Classifier

Overview

This project is a machine learning-based spam email classifier that detects spam messages using Natural Language Processing (NLP) techniques. The model is trained using TF-IDF vectorization and a Naïve Bayes classifier with data balancing via SMOTE.

Features

Preprocessing of email text data

TF-IDF feature extraction with bigrams

Handling class imbalance using SMOTE

Model training using Naïve Bayes classifier

Performance evaluation with accuracy, classification report, and confusion matrix visualization

Real-time spam detection on custom input

Installation

Clone the repository:

git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

Install dependencies:

pip install -r requirements.txt

Usage

Place your dataset (mail_data.csv) in the project folder.

Run the script:

python spam_classifier.py

To test a custom email, modify the input_mail variable in the script.

Dataset

The dataset used should be a CSV file with two columns:

Category: Label (spam or ham)

Message: The email text

Model Evaluation

Accuracy Score: Measures overall correctness

Classification Report: Precision, recall, and F1-score breakdown

Confusion Matrix: Visualizes false positives/negatives

Example Prediction

Input:

"Congratulations! You have won a free trip to the Bahamas. Click the link below to claim your prize!"

Output:

Spam

