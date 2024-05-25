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
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


## load the model
nlpmodel=pickle.load(open('ovr_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    new_data= tfidf_vectorizer.transform(data.values())
    output=nlpmodel.predict(new_data)
    return jsonify(output.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data=request.form['data']
    new_data=tfidf_vectorizer.transform(data.values())
    output=nlpmodel.predict(new_data)
    return render_template('home.html', prediction = output.tolist())

if __name__ == '__main__':
    app.run(debug=True)

