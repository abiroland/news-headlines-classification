
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
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import ZeroShotClassification
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP

# process data
# defining function that contains punctuation removal
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

# Topic Output

def target(data):
    categories = {
        0: "Technology",
        1: "Sports",
        2: "Education",
        3: "Entertainment",
        4: "Business"
    }
    return categories.get(data, "Not in database")

candidate_topics = ['business', 'politics', 'sports', 'health', 'technology', 'entertainment', 'science', 'world', 'economy', 'education']
representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 1))

# Load the model
# Flask app
app=Flask(__name__)
# Define the models
bertmodel = BERTopic.load("topic_model", embedding_model=embedding_model)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.values()
    punt_removed = remove_punctuation(data)  
    convt_lower = punt_removed.lower()
    topics, probs = bertmodel.transform(convt_lower)
    new_predictions = pd.DataFrame({"topic_labels": [target(topic) for topic in topics]})
    return render_template("home.html", prediction = new_predictions['topic_labels'].values[0])


@app.route('/predict_api', methods=['POST'])
def predict_app():
    data = request.json['data']
    punt_removed = remove_punctuation(list(data.values())[0])  
    convt_lower = punt_removed.lower()
    topics, probs = bertmodel.transform(convt_lower)
    new_predictions = pd.DataFrame({"topic_labels": [target(topic) for topic in topics]})
    return jsonify(new_predictions["topic_labels"].tolist())      


if __name__ == '__main__':
    app.run(debug=True)

