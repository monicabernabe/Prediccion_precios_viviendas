# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split

# Cargar datos y dividir el conjunto de datos
def load_and_split_data(data):
    try:
        df_precio = pd.read_csv(data)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'HousingPrices.csv'.")
        sys.exit()
    # Dividir el conjunto de datos
    train_data, test_data = train_test_split(df_precio, test_size=0.2, random_state=42)
    return train_data, test_data
