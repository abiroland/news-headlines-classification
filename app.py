
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import joblib



app=Flask(__name__)
# process data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = joblib.load('transformer_model.joblib')


## load the model
nlpmodel=joblib.load('ovr_model.joblib')

def targetname(x):
    if x == 0:
        return "Business"
    elif x == 1:
        return "Education"
    elif x == 2:
        return "Entertainment"
    elif x == 3:
        return "Sport"
    elif x == 4:
        return "Technology"
    else:
        return "Unknown"


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    new_data= tfidf_vectorizer.transform(data) 
    output=nlpmodel.predict(new_data)
    return jsonify(output.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data=request.form['data']
    new_data=tfidf_vectorizer.transform(data)
    output=nlpmodel.predict(new_data)
    return render_template('home.html', prediction = output.tolist())

if __name__ == '__main__':
    app.run(debug=True)

