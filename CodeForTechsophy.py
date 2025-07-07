import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ------------------ Preprocessing ------------------ #
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# ------------------ Model Training ------------------ #
def train_model(X, y, model_type='nb'):
    if model_type == 'nb':
        clf = MultinomialNB()
    elif model_type == 'svm':
        clf = SVC(kernel='linear', probability=True)
    else:
        raise ValueError("Unsupported model type.")

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    pipe.fit(X, y)
    return pipe

# ------------------ Load & Train ------------------ #
def prepare_and_train_models():
    # Example dataset
    data = pd.DataFrame({
        'email': [
            "Meeting at 3pm with HR team.",
            "50% off on all products, limited time offer!",
            "Urgent: server down, please check immediately.",
            "Your invoice is attached",
            "Join our webinar on sales techniques",
            "Free rewards waiting for you, click here!",
            "Please schedule an interview with the candidate",
            "Internal audit report submitted",
            "Win an iPhone today!"
        ],
        'spam': [0, 1, 0, 0, 0, 1, 0, 0, 1],
        'priority': ['low', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'low'],
        'department': ['HR', 'Marketing', 'IT', 'Finance', 'Sales', 'Marketing', 'HR', 'Finance', 'Marketing']
    })

    data['email'] = data['email'].apply(preprocess_text)

    # Encode labels
    le_priority = LabelEncoder()
    le_department = LabelEncoder()
    data['priority_encoded'] = le_priority.fit_transform(data['priority'])
    data['department_encoded'] = le_department.fit_transform(data['department'])

    # Train spam model
    spam_model = train_model(data['email'], data['spam'], model_type='nb')

    # Train priority model
    priority_model = train_model(data['email'], data['priority_encoded'], model_type='svm')

    # Train department model
    dept_model = train_model(data['email'], data['department_encoded'], model_type='svm')

    # Save all
    pickle.dump(spam_model, open('spam_model.pkl', 'wb'))
    pickle.dump(priority_model, open('priority_model.pkl', 'wb'))
    pickle.dump(dept_model, open('dept_model.pkl', 'wb'))
    pickle.dump(le_priority, open('le_priority.pkl', 'wb'))
    pickle.dump(le_department, open('le_department.pkl', 'wb'))

    print("✅ Models trained and saved.")

# ------------------ Prediction ------------------ #
def classify_email(email_text):
    email_text = preprocess_text(email_text)

    # Load models
    spam_model = pickle.load(open('spam_model.pkl', 'rb'))
    priority_model = pickle.load(open('priority_model.pkl', 'rb'))
    dept_model = pickle.load(open('dept_model.pkl', 'rb'))
    le_priority = pickle.load(open('le_priority.pkl', 'rb'))
    le_department = pickle.load(open('le_department.pkl', 'rb'))

    # Predict
    spam_pred = spam_model.predict([email_text])[0]
    spam_prob = spam_model.predict_proba([email_text])[0][spam_pred]

    priority_pred = priority_model.predict([email_text])[0]
    dept_pred = dept_model.predict([email_text])[0]

    priority_label = le_priority.inverse_transform([priority_pred])[0]
    dept_label = le_department.inverse_transform([dept_pred])[0]

    result = {
        "Spam": "Spam" if spam_pred == 1 else "Ham",
        "Spam Confidence": round(spam_prob * 100, 2),
        "Priority": priority_label,
        "Department": dept_label
    }

    return result

# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Smart Email Classifier')
    parser.add_argument('--train', action='store_true', help='Train and save models')
    parser.add_argument('--email', type=str, help='Classify a given email string')

    args = parser.parse_args()

    if args.train:
        prepare_and_train_models()

    elif args.email:
        result = classify_email(args.email)
        print("\n📩 Email Classification Result:")
        for key, value in result.items():
            print(f"{key}: {value}")

    else:
        print("❗ Use --train to train models or --email \"your email text\" to classify.")

