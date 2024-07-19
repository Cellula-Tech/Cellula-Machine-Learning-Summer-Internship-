from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from joblib import load
import numpy as np
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the scaler and the model
scaler = load('scaler.joblib')
model = load('best_random_forest_model.joblib')

market_segment_map = {
    'Offline': 1,
    'Online': 0
}

# Define expected columns based on preprocessing
expected_columns = ['lead time', 'average price', 'special requests', 'day', 'month', 
                  'number of weekend nights', 'number of week nights', 'Online', 
                  'number of adults', 'year', 'Offline']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_prelight_response()

    try:
        data = request.get_json()

        # Convert categorical variables to numerical values
        market_segment = market_segment_map.get(data['marketSegment'], 0)

        # Prepare the features for prediction
        features = {
            'number of adults': int(data['numAdults']),
            'number of weekend nights': int(data['numWeekendNights']),
            'number of week nights': int(data['numWeekNights']),
            'lead time': int(data['leadTime']),
            'average price': float(data['averagePrice']),  # Make sure this matches what model expects
            'special requests': int(data['specialRequests']),
            'day': int(datetime.fromisoformat(data['reservationDate']).day),
            'month': int(datetime.fromisoformat(data['reservationDate']).month),
            'year': int(datetime.fromisoformat(data['reservationDate']).year),
            'Offline': 1 if data['marketSegment'] == 'Offline' else 0,
            'Online': 1 if data['marketSegment'] == 'Online' else 0
        }

        # Create a DataFrame from features
        features_df = pd.DataFrame([features])

        # Reorder columns to match expected order
        features_df = features_df[expected_columns]

        # Scale the features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_result = "Canceled" if prediction == 1 else "Not Canceled"

        return jsonify({"prediction": prediction_result})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400


def _build_cors_prelight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
