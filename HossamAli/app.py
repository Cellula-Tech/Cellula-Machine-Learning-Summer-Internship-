from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            int_features  = [int(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            probabilities = model.predict_proba(final_features)[0]

            cancel_prob = probabilities[1] * 100  # Probability of cancellation
            not_cancel_prob = probabilities[0] * 100  # Probability of not cancellation

            result = f"Probability of Cancellation: {cancel_prob:.2f}%, Probability of Not Cancellation: {not_cancel_prob:.2f}%"
            return render_template('index.html', prediction_text=f"Prediction: {result}")
        
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)