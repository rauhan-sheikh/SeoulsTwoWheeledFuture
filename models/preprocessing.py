import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def date_tf(XX):
    XX_copy = XX.copy()  # Create a copy to avoid modifying the original dataframe
    XX_copy['Date'] = pd.to_datetime(XX_copy['Date'], format='%d/%m/%Y')
    Day = XX_copy['Date'].dt.day
    Month = XX_copy['Date'].dt.month
    Year = XX_copy['Date'].dt.year

    dmy = pd.DataFrame({'Day': Day, 'Month': Month, 'Year': Year})

    return dmy

def day_name_tf(XX):
    Day_Name = pd.DataFrame({'Day Name':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    ohe = OneHotEncoder(sparse_output=False, drop='first').fit(Day_Name)
    
    XX_copy = XX.copy()
    XX_copy['Date'] = pd.to_datetime(XX_copy['Date'],format='%d/%m/%Y')
    dn = pd.DataFrame(XX_copy['Date'].dt.day_name().values,columns=['Day Name'])
    dn = pd.DataFrame(ohe.transform(dn))
    return dn
    