from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from libraries import *

app = Flask(__name__)

# Load pipeline and model
pipeline = joblib.load('./FLASK Deploy/models/pipeline.joblib')
model = joblib.load('./FLASK Deploy/models/model.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Booking_ID': [request.form.get('Booking_ID')],
        'number of adults': [request.form.get('number of adults')],
        'number of children': [request.form.get('number of children')],
        'number of weekend nights': [request.form.get('number of weekend nights')],
        'number of week nights': [request.form.get('number of week nights')],
        'type of meal': [request.form.get('type of meal')],
        'car parking space': [request.form.get('car parking space')],
        'room type': [request.form.get('room type')],
        'lead time': [request.form.get('lead time')],
        'market segment type': [request.form.get('market segment type')],
        'repeated': [request.form.get('repeated')],
        'P-C': [request.form.get('P-C')],
        'P-not-C': [request.form.get('P-not-C')],
        'average price': [request.form.get('average price')],
        'special requests': [request.form.get('special requests')],
        'date of reservation': [request.form.get('date of reservation')],
    }

    new_data = pd.DataFrame(data)
    new_data = new_data.astype({
        'number of adults': 'int',
        'number of children': 'int',
        'number of weekend nights': 'int',
        'number of week nights': 'int',
        'car parking space': 'int',
        'lead time': 'int',
        'repeated': 'int',
        'P-C': 'int',
        'P-not-C': 'int',
        'average price': 'float',
        'special requests': 'int'
    })
    transformed_data = pipeline.transform(new_data)
    prediction = model.predict(transformed_data)
    prediction_text = 'Confirmed' if prediction[0] == 1 else 'Not Confirmed'
    return jsonify({'prediction_text': prediction_text})


if __name__ == '__main__':
    app.run(debug=True)
