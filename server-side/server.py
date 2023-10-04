from flask import Flask, request, render_template, url_for, redirect
from flask_cors import CORS

import requests
import json

from modelling.train_model import MultivariateLinearRegression

app = Flask(__name__)

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://127.0.0.1:5500", "http://127.0.0.1:5173"])

@app.route('/', methods=['GET'])
def input_form():
    return render_template('index.html', title='Housing values')


@app.route('/predict', methods=['POST'])
@app.route('/predict/json', methods=['POST'])
def get_prediction():
    model = MultivariateLinearRegression()
    model.load_weights('./modelling/weights/coefficients.json')

    raw_data = request.form
    longitude = raw_data['longitude']
    latitude = raw_data['latitude']
    house_med_age = raw_data['housing-median-age']
    total_rooms = raw_data['total-rooms']
    total_bedrooms = raw_data['total-bedrooms']
    population = raw_data['population']
    households = raw_data['households']
    med_income = raw_data['median-income']

    print(f"{longitude}, {latitude}, {house_med_age}, {total_rooms}, {total_bedrooms}, {population}, {households}, {med_income}")

    if "/json" in request.path:
        # return instead json format of 
        # predictions for the front end
        pass

    return redirect(url_for('input_form'))

if __name__ == "__main__":
    app.run(debug=True)
    

