from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
scaler = joblib.load('D:/Python/CellulaFinal/scaler.pkl')
# Load your trained machine learning model
your_model = joblib.load('D:/Python/CellulaFinal/ensemble_model.pkl')

# Initialize MinMaxScaler


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Assuming JSON input from your React frontend
        
        # Extract data from JSON and convert to a list of integers and floats
        input_data = [np.int64(data['input1']), np.int64(data['input2']), np.int64(data['input3']), np.int64(data['input4']),
                  np.int64(data['input5']), np.int64(data['input6']), np.int64(data['input7']), np.int64(data['input8']),
                  np.int64(data['input9']), np.float64(data['input10']), np.int64(data['input11']), np.int64(data['input12'])]
    
        feature_names = [
    'number of adults', 'number of children', 'number of weekend nights',
    'number of week nights', 'type of meal', 'car parking space', 'room type',
    'lead time', 'market segment type', 'average price', 'special requests',
    'date of reservation'
]
        # Reshape input_data to fit MinMaxScaler's expected shape
        input_data_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale the input data using MinMaxScaler
        input_data_scaled = scaler.transform(input_data_df)
        
        
        # Predict class probabilities
        probabilities = your_model.predict_proba(input_data_scaled)[0]
        
        # Predict class label
        predicted_class = your_model.predict(input_data_scaled)[0]
        
        # Prepare response JSON
        wordd='Canceled' 
        if predicted_class ==1:
          wordd='Not_Canceled'
          

        response = {
            'predicted_class': wordd,  # Convert predicted_class to string
            'probabilities': probabilities.tolist()  # Convert numpy array to list for JSON serialization
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})  # Return error message in case of any exception

if __name__ == '__main__':
    app.run(debug=True)
