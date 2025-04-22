from flask import Flask, request, jsonify
import joblib
from model import get_weather_data, detect_cropping_season
import numpy as np

app = Flask(__name__)

# Load your trained model and encoders
model = joblib.load('crop_model.pkl')
season_encoder = joblib.load('season_encoder.pkl')
rainfed_encoder = joblib.load('rainfed_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')

@app.route('/')
def index():
    return "running"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the request or use default values

    data = request.get_json()
    print(data)

    field_size = data['field_size']
    lat = data['lat']
    lon = data['lon']
    rainfed = data['rainfed']


    # field_size = 4
    # lat, lon = 21.1458, 79.0882
    season = detect_cropping_season()
    # rainfed = 'Yes'

    # Get weather data
    temp_min, temp_max = get_weather_data(lat, lon)

    # Encode categorical variables
    season_encoded = season_encoder.transform([season])[0]
    rainfed_encoded = rainfed_encoder.transform([rainfed])[0]

    # Prepare input data for prediction
    input_data = [[field_size, season_encoded, temp_min, temp_max, rainfed_encoded]]
    crop_encoded = model.predict(input_data)[0]

    # Decode the crop prediction
    crop_name = crop_encoder.inverse_transform([crop_encoded])[0]

    return jsonify({'recommended_crop': str(crop_name)})

    # Make prediction
    prediction = model.predict(input_data)
    print(prediction)

    # Convert prediction to native Python type
    recommended_crop = prediction[0]
    if isinstance(recommended_crop, np.generic):
        recommended_crop = recommended_crop.item()

    # Return the prediction as JSON
    return jsonify({'recommended_crop': recommended_crop})

if __name__ == '__main__':
    app.run(debug=True)
