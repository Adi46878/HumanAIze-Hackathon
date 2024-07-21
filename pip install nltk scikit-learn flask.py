pip install nltk scikit-learn flask
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

questions = ["What are your working hours?", "How can I reset my password?"]
answers = ["Our working hours are from 9 AM to 6 PM, Monday to Friday.", "You can reset your password by clicking on 'Forgot Password' on the login page."]

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)
y = answers

model = MultinomialNB()
model.fit(X, y)
from flask import Flask, request, jsonify

app = Flask(_name_)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json['question']
    processed_input = preprocess(user_input)
    transformed_input = vectorizer.transform([processed_input])
    response = model.predict(transformed_input)[0]
    return jsonify({"answer": response})

if _name_ == '_main_':
    app.run(debug=True)