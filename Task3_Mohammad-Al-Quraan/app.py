from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    model_path = 'best_model_selected_features.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            type_of_meal = int(request.form['type_of_meal'])
            average_price = int(request.form['average_price'])
            number_of_week_nights = int(request.form['number_of_week_nights'])
            number_of_weekend_nights = int(request.form['number_of_weekend_nights'])
            room_type = int(request.form['room_type'])
            lead_time = int(request.form['lead_time'])
            special_requests = int(request.form['special_requests'])
            year = int(request.form['year'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            
            features = np.array([[type_of_meal, average_price, number_of_week_nights, number_of_weekend_nights, room_type, lead_time, special_requests, year, month, day]])
            probabilities = model.predict_proba(features)[0]

            cancel_prob = probabilities[1] * 100  # Probability of cancellation
            not_cancel_prob = probabilities[0] * 100  # Probability of not cancellation

            result = f"Probability of Cancellation: {cancel_prob:.2f}%, Probability of Not Cancellation: {not_cancel_prob:.2f}%"

            return render_template('index.html', prediction_text=f"Prediction: {result}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
