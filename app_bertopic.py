
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
import safetensors
import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
import os
from dotenv import load_dotenv

# process data
# defining function that contains punctuation removal
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# Topic Output

def target(data):
    if data == 0:
        return "Entertainment"
    elif data == 1:
        return "Education"
    elif data == 2:
        return "Technology"
    elif data == 3:
        return "Business"
    elif data == 4:
        return "Sports"
    else:
        return "Unknown"


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Load the model
# Flask app
app=Flask(__name__)
# Load the model
client = openai.OpenAI(api_key=api_key)
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 1))

# Define the models
representation_model = OpenAI(client, model="gpt-3.5-turbo", chat=True)
model = BERTopic.load("topic_model")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data=request.form.values()
    punt_removed=remove_punctuation(data)  
    convt_lower=punt_removed.lower()
    output=model.transform(convt_lower)
    return render_template("home.html", prediction = target(output[0]))


@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    punt_removed=remove_punctuation(list(data.values())[0])  
    convt_lower=punt_removed.lower()
    output=model.transform(convt_lower)
    return jsonify(target(output[0].tolist()))      


if __name__ == '__main__':
    app.run(debug=True)

