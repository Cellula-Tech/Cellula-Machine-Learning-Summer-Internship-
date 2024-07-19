from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# This should match the features used during training
selected_features = [
    'lead time',
    'average price',
    'special requests',
    'type of meal_Meal Plan 2',
    'market segment type_Corporate'
]

@app.route('/')
def home():
    return render_template('index.html', features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features = [float(data[feature]) for feature in selected_features]
        features = np.array([features])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        result = 'Canceled' if prediction[0] == 0 else 'Not Canceled'
        return render_template('index.html', features=selected_features, prediction=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
