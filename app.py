from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Conversion factor from MPG to KM/L
MPG_TO_KPL = 0.425144

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data from the POST request
    features = np.array(data['features']).reshape(1, -1)  # Convert data to numpy array and reshape for the model

    # Make prediction
    prediction_mpg = model.predict(features)[0]  # Get the prediction value in MPG

    # Convert the prediction from MPG to KM/L
    prediction_kpl = prediction_mpg * MPG_TO_KPL

    # Define a threshold for good efficiency in KM/L
    threshold_kpl = 25 * MPG_TO_KPL  # Convert 25 MPG to KM/L

    if prediction_kpl >= threshold_kpl:
        efficiency = 'Good'
    else:
        efficiency = 'Not Efficient'

    # Return the prediction and efficiency as a JSON response
    return jsonify({'prediction': prediction_kpl, 'efficiency': efficiency})

if __name__ == '__main__':
    app.run(debug=True)
