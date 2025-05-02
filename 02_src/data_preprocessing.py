# Importar librerías
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from cargar_datos import load_and_split_data

# Cargar los datos
train_data, test_data = load_and_split_data('data/HousingPrices.csv')

# Funcion para limpieza de datos
def cleaning_data (data):
    data = data.iloc[:,3:]                    # Eliminación de columnas innecesarias
    data = data.dropna()                      # Eliminación de valores nulos
 
train_data = cleaning_data (train_data)
test_data = cleaning_data (test_data)

# Extracción de la variable objetivo
y_train = train_data['Price']
y_test = test_data['Price']

# Eliminación de la variable objetivo del DataFrame
X_train = train_data.drop('Price', axis=1)
X_test = test_data.drop('Price', axis=1)