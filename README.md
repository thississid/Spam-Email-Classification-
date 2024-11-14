# Spam Email Classifier

This is a Flask-based web application that uses a machine learning model to classify email text as "Spam" or "Not Spam". The app displays model performance metrics like accuracy, precision, recall, and F1-score, and allows users to enter email text for live classification.

## Features
- **Spam Classification**: Classifies emails as "Spam" or "Not Spam" based on the input text.
- **Performance Metrics**: Displays metrics such as accuracy, precision, recall, and F1-score on the home page.
- **Interactive UI**: Clean and colorful UI with text area input and classification results.

## Project Structure
```plaintext
.
├── app.py                 # Flask application code
├── spam_classifier_model.pkl  # Trained model file
├── vectorizer.pkl         # Vectorizer for transforming text data
├── metrics.pkl            # Pickle file with model performance metrics
├── static/
│   └── styles.css         # CSS file for styling the UI
├── templates/
│   ├── index.html         # Home page with metrics and input form
│   └── result.html        # Result page displaying classification result
├── spam_ham_dataset.csv   # Dataset file used for model training
└── README.md              # Project documentation

## Setup Instructions

### Prerequisites
- Python 3.x
- Flask
- scikit-learn
- pandas

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thississid/Spam-Email-Classification-.git
   cd spam-email-classifier
   python app.py
