from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:/Users/DELL/final_model.pkl')

# Load the selected features
selected_features = joblib.load('C:/Users/DELL/selected_features.pkl')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Convert JSON data to DataFrame
    input_data = pd.DataFrame([data])

    # Ensure the order of columns matches the training data
    input_data = input_data[selected_features]

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction as JSON
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
