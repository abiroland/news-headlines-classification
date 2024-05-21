import tf_keras
import keras
import tensorflow as tf
from keras.models import load_model
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
# process data
def fultrn(preds):

    from tf_keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    maxlen = 100
    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(preds)
    sequences = tokenizer.texts_to_sequences(preds)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=maxlen)
    return data


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


## load the model
nlpmodel=load_model('model.keras')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_app():
    data=request.json['data']
    new_data=np.array(fultrn(data.values()))
    output=np.argmax(nlpmodel.predict(new_data), axis=1)
    return jsonify(targetnames(output))

if __name__ == '__main__':
    app.run(debug=True)

