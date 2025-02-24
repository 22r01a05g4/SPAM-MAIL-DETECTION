# Spam Email Classifier

## Overview

This project is a machine learning-based spam email classifier that detects spam messages using Natural Language Processing (NLP) techniques. The model is trained using TF-IDF vectorization and a Naïve Bayes classifier with data balancing via SMOTE.

## Features

- Preprocessing of email text data
- TF-IDF feature extraction with bigrams
- Handling class imbalance using SMOTE
- Model training using Naïve Bayes classifier
- Performance evaluation with accuracy, classification report, and confusion matrix visualization
- Real-time spam detection on custom input

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-classifier.git
   cd spam-classifier
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Place your dataset (`mail_data.csv`) in the project folder.
2. Run the script:
   ```bash
   python spam_classifier.py
3. To test a custom email, modify the `input_mail` variable in the script.
   
## Dataset
The dataset used should be a CSV file with two columns:
- Category: Label (spam or ham)
- Message: The email text

## Model Evaluation
- Accuracy Score: Measures overall correctness
- Classification Report: Precision, recall, and F1-score breakdown
- Confusion Matrix: Visualizes false positives/negatives
  
- **Training Accuracy**: 98.65%  
- **Testing Accuracy**: 98.21%  

## Example Prediction
Input:

    "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"

Output:
    
    Spam Mail
