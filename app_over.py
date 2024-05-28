from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import pickle



app=Flask(__name__)
# process data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = pickle.load(open('transformer_model.pkl', 'rb'))


## load the model
nlpmodel=pickle.load(open('ovr_model.pkl', 'rb'))

def targetnames(result):
    if result == 0:
        return "Business"
    elif result == 1:
        return "Education"
    elif result == 2:
        return "Entertainment"
    elif result == 3:
        return "Sports"
    elif result == 3:
        return "Technology"
    else:
        return "Unknown"


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    data = [data]
    new_data= tfidf_vectorizer.transform(data) 
    output=nlpmodel.predict(new_data)
    return jsonify(targetnames(output))

@app.route('/predict', methods=['POST'])
def predict():
    if 'data' not in request.form:
        return jsonify({'error': 'Missing data field'}), 400
    data=request.form['data']
    data = [data]
    new_data=tfidf_vectorizer.transform(data)
    output=nlpmodel.predict(new_data)
    return render_template('home.html', prediction = targetnames(output))

if __name__ == '__main__':
    app.run(debug=True)

