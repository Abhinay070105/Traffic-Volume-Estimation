import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('G:/AIML/ML projects/Traffic_volume/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']]
    data = pandas.DataFrame(features_values, columns=names)
    data = scale.fit_transform(data)
    data = pandas.DataFrame(data, columns=names)
    prediction = model.predict(data)
    text = "Estimated Traffic Volume is : "
    return render_template('index.html', prediction_text=text + str(prediction))

if __name__ == "__main__":
    app.run(port=8000, debug=True, use_reloader=False)
