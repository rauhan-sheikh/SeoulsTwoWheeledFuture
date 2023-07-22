from flask import Flask, render_template, request
import pandas as pd
from models.preprocessing import date_tf, day_name_tf
import pickle


app = Flask(__name__)


class BikeRentalPredictor:
    def __init__(self, model_path):
        self.pipe = pickle.load(open(model_path, 'rb'))

    def predict(self, input_data):
        predictions = self.pipe.predict(input_data)
        return predictions

# Load the trained model
model_path = 'models/xgb_regressor_pipeline_r2_0_932_v2.pkl'
bike_rental_predictor = BikeRentalPredictor(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        date = request.form['date']
        hour = int(request.form['hour'])
        temperature = float(request.form['temperature'])
        humidity = int(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        visibility = int(request.form['visibility'])
        dew_point = float(request.form['dew_point'])
        solar_radiation = float(request.form['solar_radiation'])
        rainfall = float(request.form['rainfall'])
        snowfall = float(request.form['snowfall'])
        seasons = request.form['seasons']
        holiday = request.form['holiday']
        functioning_day = request.form['functioning_day']

        # Create a dictionary with the user input data
        user_input = {
            'Date': [date],
            'Hour': [hour],
            'Temperature(°C)': [temperature],
            'Humidity(%)': [humidity],
            'Wind speed (m/s)': [wind_speed],
            'Visibility (10m)': [visibility],
            'Dew point temperature(°C)': [dew_point],
            'Solar Radiation (MJ/m2)': [solar_radiation],
            'Rainfall(mm)': [rainfall],
            'Snowfall (cm)': [snowfall],
            'Seasons': [seasons],
            'Holiday': [holiday],
            'Functioning Day': [functioning_day]
        }

        # Convert the user input dictionary to a DataFrame
        input_data = pd.DataFrame(user_input)

        # Make predictions using the model
        predictions = bike_rental_predictor.predict(input_data)

        return render_template('index.html', predictions=round(predictions[0]))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
