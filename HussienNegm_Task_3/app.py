import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Create Flask app
flask_app = Flask(__name__)

# Ensure the working directory is the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.pkl')

# Load the model
model = pickle.load(open(model_path, 'rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Extract features from the form
    float_features = [
        float(request.form['number_of_weekend_nights']),
        float(request.form['number_of_week_nights']),
        float(request.form['lead_time']),
        float(request.form['market_segment_type']),
        float(request.form['average_price']),
        float(request.form['special_requests'])
    ]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]

    if prediction == 0:
        prediction_text = "The booking status is canceled"
    else:
        prediction_text = "The booking status is not canceled"
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
