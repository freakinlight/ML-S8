import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model_filename = "model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ECG Classification API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data["features"]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return prediction
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

