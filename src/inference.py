import pickle
import pandas as pd
#from BikeSharingDemand.models.preprocessing import date_tf, day_name_tf
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def date_tf(XX):
    XX_copy = XX.copy()  # Create a copy to avoid modifying the original dataframe
    XX_copy['Date'] = pd.to_datetime(XX_copy['Date'], format='%d/%m/%Y')
    Day_Number = pd.Categorical(XX_copy['Date'].dt.day)
    Month = XX_copy['Date'].dt.month_name()

    day_number_df = pd.DataFrame({'Day Number': Day_Number})
    month_df = pd.DataFrame({'Month': Month})

    dn_df = pd.DataFrame({'Day Number': [i for i in range(1, 32)]})
    m_df = pd.DataFrame({'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                   'September', 'October', 'November', 'December']})

    dn_ohe = OneHotEncoder(sparse_output=False, drop='first').fit(dn_df)
    m_ohe = OneHotEncoder(sparse_output=False, drop='first').fit(m_df)

    dn_transformed = pd.DataFrame(dn_ohe.transform(day_number_df))
    m_transformed = pd.DataFrame(m_ohe.transform(month_df))
    dm = pd.concat([dn_transformed, m_transformed], axis=1)

    return dm


def day_name_tf(XX):
    Day_Name = pd.DataFrame(
        {'Day Name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    ohe = OneHotEncoder(sparse_output=False, drop='first').fit(Day_Name)

    XX_copy = XX.copy()
    XX_copy['Date'] = pd.to_datetime(XX_copy['Date'], format='%d/%m/%Y')
    dn = pd.DataFrame(XX_copy['Date'].dt.day_name().values, columns=['Day Name'])
    dn = pd.DataFrame(ohe.transform(dn))
    return dn

    

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

    model_path = r'models/xgb_regressor_pipeline_r2_0_932_v2.pkl'
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
