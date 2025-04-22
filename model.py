import pandas as pd
data = {
    'field_size': [3]*35,   # example: assuming all fields have size 3, you can adjust this later if you like
    'season': [
        'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Kharif', 'Kharif', 'Kharif', 'Kharif', 'Kharif', 'Kharif',
        'Zaid', 'Rabi', 'Rabi', 'Kharif', 'Rabi', 'Rabi', 'Rabi', 'Kharif', 'Rabi', 'Rabi',
        'Kharif', 'Zaid', 'Kharif', 'Zaid', 'Zaid', 'Zaid', 'Rabi', 'Rabi', 'Rabi', 'Kharif',
        'Kharif', 'Kharif', 'Kharif', 'Rabi', 'Rabi'
    ],
    'temp_min': [
        25, 15, 20, 10, 20, 20, 20, 20, 20, 21,
        20, 20, 10, 20, 15, 20, 15, 20, 15, 15,
        20, 25, 24, 15, 10, 15, 10, 15, 15, 20,
        25, 20, 25, 10, 8
    ],
    'temp_max': [
        35, 25, 30, 20, 35, 35, 30, 30, 30, 30,
        40, 30, 25, 35, 25, 30, 25, 30, 20, 20,
        30, 35, 30, 35, 30, 25, 18, 25, 25, 30,
        35, 35, 35, 25, 20
    ],
    'rainfed': [
        'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
        'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No',
        'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes'
    ],
    'crop_name': [
        'Rice', 'Wheat', 'Maize', 'Barley', 'Sorghum', 'Pearl Millet', 'Ragi', 'Groundnut', 'Soybean', 'Cotton',
        'Sugarcane', 'Sunflower', 'Mustard', 'Sesame', 'Potato', 'Tomato', 'Onion', 'Chillies', 'Cauliflower', 'Cabbage',
        'Brinjal', 'Banana', 'Mango', 'Grapes', 'Tea', 'Coffee', 'Peas', 'Chickpea', 'Lentil', 'Pigeon Pea',
        'Black Gram', 'Green Gram', 'Cowpea', 'Spinach', 'Carrot'
    ]
}

df = pd.DataFrame(data)


from sklearn.preprocessing import LabelEncoder

season_encoder = LabelEncoder()
rainfed_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df['season'] = season_encoder.fit_transform(df['season'])
df['rainfed'] = rainfed_encoder.fit_transform(df['rainfed'])
df['crop_name'] = crop_encoder.fit_transform(df['crop_name'])


from sklearn.ensemble import RandomForestClassifier

X = df[['field_size', 'season', 'temp_min', 'temp_max', 'rainfed']]
y = df['crop_name']

model = RandomForestClassifier()
model.fit(X, y)

def predict_crop(field_size, season, temp_min, temp_max, rainfed):
    # Encode text inputs
    season_encoded = season_encoder.transform([season])[0]
    rainfed_encoded = rainfed_encoder.transform([rainfed])[0]

    input_data = [[field_size, season_encoded, temp_min, temp_max, rainfed_encoded]]
    crop_encoded = model.predict(input_data)[0]

    # Decode the crop name
    crop_name = crop_encoder.inverse_transform([crop_encoded])[0]
    return crop_name

import requests

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_min,temperature_2m_max&timezone=auto"
    response = requests.get(url)
    data = response.json()

    temp_min = data['daily']['temperature_2m_min'][0]
    temp_max = data['daily']['temperature_2m_max'][0]

    return temp_min, temp_max

from datetime import datetime


def detect_cropping_season():
    month = datetime.now().month

    if 3 <= month <= 6:
        return "Zaid"
    elif 6 <= month <= 10:
        return "Kharif"
    else:
        return "Rabi"


# lat, lon = 11.0168, 76.9558
# temp_min, temp_max  = get_weather_data(lat, lon)
#
# # Now predict
# result = predict_crop(field_size=1.5, season=detect_cropping_season(), temp_min=temp_min, temp_max=temp_max, rainfed='No')
# print("Recommended Crop:", result)

import joblib

# Assuming 'model' is your trained model
joblib.dump(model, 'crop_model.pkl')
joblib.dump(season_encoder, 'season_encoder.pkl')
joblib.dump(rainfed_encoder, 'rainfed_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')


