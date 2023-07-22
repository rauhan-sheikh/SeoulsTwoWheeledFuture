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

    