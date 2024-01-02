from flask import Flask, request, render_template, url_for, redirect, jsonify
from flask_cors import CORS

import requests
import json

from modelling.train_model import MultivariateLinearRegression
import numpy as np

app = Flask(__name__)

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://127.0.0.1:5500", "http://127.0.0.1:5173"])

# global variable
model = None

def load_model():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # redeclare as global model
    global model

    # instantiate model and load weights
    model = MultivariateLinearRegression()
    model.load_weights('./modelling/weights/meta_data.json')

# load model upon start of server
load_model()



@app.route('/', methods=['GET'])
def input_form():
    return render_template('index.html', title='Housing values')

@app.route('/predict', methods=['POST'])
@app.route('/predict/json', methods=['POST'])
def get_prediction():

    # get raw data and extract individual values
    raw_data = request.form

    # min is -124 and max is -114
    longitude = float(raw_data['longitude'])

    # min is 32.5 and max is 42
    latitude = float(raw_data['latitude'])

    # 1 and 52
    house_med_age = float(raw_data['housing-median-age'])

    # 1 and 39300
    total_rooms = float(raw_data['total-rooms'])

    # 1 and 6450
    total_bedrooms = float(raw_data['total-bedrooms'])

    # 3 and 35700
    population = float(raw_data['population'])

    # 1 and 6080
    households = float(raw_data['households'])

    # 0.5 and 15
    med_income = float(raw_data['median-income'])

    print(f"{longitude}, {latitude}, {house_med_age}, {total_rooms}, {total_bedrooms}, {population}, {households}, {med_income}")
    X = np.array([longitude, latitude, house_med_age, total_rooms, total_bedrooms, population, households, med_income])

    # predict given values
    Y_pred = model.predict(X)

    if "/json" in request.path:
        # return instead json format of 
        # predictions for the front end
        return jsonify({'median-house-value': Y_pred})

    return redirect(url_for('input_form'))

    

