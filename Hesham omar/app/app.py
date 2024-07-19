from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    total_people = int(request.form['total_people'])
    total_nights = int(request.form['total_nights'])
    date_of_reservation = request.form['date_of_reservation']
    lead_time = int(request.form['lead_time'])
    average_price = float(request.form['average_price'])
    meal_plan = int(request.form['meal_plan'])
    room_type = int(request.form['room_type'])
    car_parking_space = int(request.form['car_parking_space'])
    special_requests = int(request.form['special_requests'])
    market_segment_type = int(request.form['market_segment_type'])

    # Process market segment type
    market_segment = [0, 0, 0, 0]  # Initialize as all zeros
    if market_segment_type == 1:
        market_segment[0] = 1  # Online
    elif market_segment_type == 2:
        market_segment[1] = 1  # Offline
    elif market_segment_type == 3:
        market_segment[2] = 1  # Corporate
    elif market_segment_type == 4:
        market_segment[3] = 1  # Complementary

    # Convert date to day, month, year
    year, month, day = map(int, date_of_reservation.split('-'))

    # Prepare features for prediction
    features = np.array([[total_people, total_nights, day, month, year, lead_time, average_price, meal_plan, room_type, car_parking_space, special_requests] + market_segment])

    # Perform prediction
    prediction = model.predict(features)[0]

    # Determine prediction text based on the model output
    if prediction == 1:
        prediction_text = "Booking is predicted to be canceled."
    else:
        prediction_text = "Booking is predicted to be non-canceled."

    # Render the form again with prediction result
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
