import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')
data = data[['text', 'label_num']]  # Assuming 'text' and 'label_num' are columns for text and label
  
# Preprocess and split the data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label_num'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save the model and vectorizer
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save metrics
metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
