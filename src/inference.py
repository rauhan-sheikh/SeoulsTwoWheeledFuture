import pickle
import pandas as pd
from models.preprocessing import date_tf, day_name_tf
from sklearn.base import BaseEstimator, TransformerMixin


class BikeRentalPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.pipe = pickle.load(open(model_path, 'rb'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.pipe.transform(X)

    def predict(self, X):
        return self.pipe.predict(X)


def main():
    model_path = r'models/xgb_regressor_pipeline_r2_0_932_v1.pkl'
    predictor = BikeRentalPredictor(model_path)

    user_input = {
        'Date': input("Enter the date (DD/MM/YYYY): "),
        'Hour': int(input("Enter the hour (0-23): ")),
        'Temperature(°C)': float(input("Enter the temperature in °C: ")),
        'Humidity(%)': int(input("Enter the humidity percentage: ")),
        'Wind speed (m/s)': float(input("Enter the wind speed in m/s: ")),
        'Visibility (10m)': int(input("Enter the visibility in meters: ")),
        'Dew point temperature(°C)': float(input("Enter the dew point temperature in °C: ")),
        'Solar Radiation (MJ/m2)': float(input("Enter the solar radiation in MJ/m2: ")),
        'Rainfall(mm)': float(input("Enter the rainfall in mm: ")),
        'Snowfall (cm)': float(input("Enter the snowfall in cm: ")),
        'Seasons': input("Enter the season (Spring, Summer, Autumn, Winter): "),
        'Holiday': input("Enter holiday status (Holiday or No Holiday): "),
        'Functioning Day': input("Enter functioning day status (Yes or No): ")
    }

    user_input_df = pd.DataFrame(user_input, index=[0])

    predictions = predictor.predict(user_input_df)
    print("Predicted bike rental count:", round(predictions[0]))


if __name__ == '__main__':
    main()
