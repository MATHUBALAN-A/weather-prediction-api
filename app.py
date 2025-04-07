from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("xgboost_weather_model.pkl", "rb"))

@app.route('/')
def home():
    return "Weather Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['temperature'], data['humidity'], data['pressure']]])
    prediction = model.predict(features)[0]
    result = "Rain" if prediction == 1 else "No Rain"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
