
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
string.punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# process data

# defining function that contains punctuation removal
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# defining function that contains tokenization
import re
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

# defining function that contains stopwords removal
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords]
    return text

# defining function that contains lemmitization
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def unlist(list):
    return " ".join(list)

from sklearn.feature_extraction.text import TfidfVectorizer

def target(data):
    if data == 0:
        return "Business"
    elif data == 1:
        return "Education"
    elif data == 2:
        return "Entertainment"
    elif data == 3:
        return "Sports"
    elif data == 4:
        return "Technology"
    else:
        return "Unknown"

# Load the model

# Flask app
app=Flask(__name__)
# Load the transformer
tfidf_vectorizer = joblib.load('transformer_model.joblib')
# Load the model
model = joblib.load('xgbmodel.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    punt_removed=remove_punctuation(list(data.values())[0])  
    convt_lower=punt_removed.lower()
    tokenized=tokenize(convt_lower)
    stop_removed=remove_stopwords(tokenized)
    lemmatized=lemmatizing(stop_removed)
    #unlisted=unlist(lemmatized)
    new_data= tfidf_vectorizer.transform(lemmatized) 
    output=model.predict(new_data)
    return jsonify(target(output[0].tolist()))      

@app.route('/predict', methods=['POST'])
def predict():
    data=request.form.values()
    punt_removed=remove_punctuation(data)  
    convt_lower=punt_removed.lower()
    tokenized=tokenize(convt_lower)
    stop_removed=remove_stopwords(tokenized)
    lemmatized=lemmatizing(stop_removed)
    #unlisted=unlist(lemmatized)
    new_data= tfidf_vectorizer.transform(lemmatized) 
    output=model.predict(new_data)
    return render_template("home.html", prediction = target(output[0]))

if __name__ == '__main__':
    app.run(debug=True)

