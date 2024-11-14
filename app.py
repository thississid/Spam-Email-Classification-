from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model, vectorizer, and metrics
with open('spam_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return render_template('result.html', result=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
