
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import joblib
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
string.punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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
tfidf_vectorizer = pickle.load(open('transformer_model.pkl','rb'))


## load the model
nlpmodel=pickle.load(open('model.pkl','rb'))


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    punt_removed=remove_punctuation(list(data))
    convt_lower=punt_removed.lower()
    tokenized=tokenize(convt_lower)
    stop_removed=remove_stopwords(tokenized)
    lemmatized=lemmatizing(stop_removed)
    unlisted=unlist(lemmatized)
    new_data= tfidf_vectorizer.transform([unlisted]) 
    output=nlpmodel.predict(new_data)
    return jsonify(output[0].tolist())      

@app.route('/predict', methods=['POST'])
def predict():
    data=request.form['data']
    punt_removed=remove_punctuation(data)  
    convt_lower=punt_removed.lower()
    tokenized=tokenize(convt_lower)
    stop_removed=remove_stopwords(tokenized)
    lemmatized=lemmatizing(stop_removed)
    #unlisted=unlist(lemmatized)
    new_data= tfidf_vectorizer.transform(lemmatized) 
    output=nlpmodel.predict(new_data)
    return render_template('home.html', prediction = output[0].tolist())

if __name__ == '__main__':
    app.run(debug=True)

