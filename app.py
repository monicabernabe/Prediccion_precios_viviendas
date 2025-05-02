import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cargar el modelo entrenado
model_path = 'GradientBoostingRegressor_model_precios_viviendas.pkl'
model = pickle.load(open(model_path, 'rb'))

# Función para realizar el preprocesamiento y la predicción
def preprocess_and_predict(area, room, lon, lat):
    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame([[area, room, lon, lat]], columns=['Area', 'Room', 'Lon', 'Lat'])
     
    # Realizar la predicción con el modelo entrenado
    price_prediction = model.predict(input_data)
    
    return price_prediction[0]

# Título de la aplicación
st.title('Predicción del Precio de la Vivienda en Ámsterdam')

# Descripción de la aplicación
st.write("""
    Esta aplicación utiliza un modelo de **Gradient Boosting Regressor** entrenado con datos sobre viviendas en Ámsterdam para predecir el precio de una vivienda en función de su área, número de habitaciones y ubicación (latitud y longitud).
""")

# Entrada para área, habitación, longitud y latitud
st.write("### Introduce los datos de la vivienda:")

# Crear los botones de entrada para cada una de las variables
area = st.number_input('Área (en metros cuadrados)')
room = st.number_input('Número de habitaciones')
lon = st.number_input('Longitud') 
lat = st.number_input('Latitud') 

# Botón para predecir el precio
if st.button('Predicción'):
    # Realizar la predicción con los datos introducidos
    predicted_price = preprocess_and_predict(area, room, lon, lat)
    
    # Mostrar el resultado de la predicción
    st.write(f'### El precio estimado de la vivienda es: {predicted_price:,.2f} €')

with st.sidebar:
    st.header("Acerca de")
    st.markdown("""
    Esta aplicación permite predecir el precio de una vivienda en función de los siguientes parámetros:

    - **Área**: entre 20 y 250 m²  
    - **Número de habitaciones**: entre 1 y 5  
    - **Longitud** (coordenada geográfica): entre 4.6 y 5.0  
    - **Latitud** (coordenada geográfica): entre 52.0 y 52.5  

    **Importante**: el modelo ha sido entrenado con datos reales y puede no ser preciso en todos los contextos.

    Si introduces valores fuera de estos rangos recomendados, los resultados pueden ser poco fiables.
    """)