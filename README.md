# Informe de Predicción del Precio de la Vivienda en Ámsterdam

## Introducción

El objetivo de este proyecto es predecir los precios de la vivienda en una zona geográfica determinada utilizando técnicas de machine learning y análisis de datos. El propósito es:

- Tomar decisiones de compra informadas.
- Identificar oportunidades de ahorro.
- Optimizar el proceso de búsqueda de vivienda en Ámsterdam.
- Aumentar la productividad de los equipos.

## Descripción de los Datos

Los datos utilizados en este proyecto provienen de Kaggle y están disponibles públicamente para su uso:

[Dataset en Kaggle](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data)

Este conjunto de datos contiene información sobre viviendas en Ámsterdam.

### Descripción de las Columnas:

- **Unnamed**:
  Indica el nº de línea del dataset.
- **Address**:  
  Dirección de la vivienda. Puede influir en el precio de la vivienda.

- **Zip**:  
  Código postal. Relacionado con la ubicación y puede influir en el precio de la vivienda.

- **Price**:  
  Precio de la vivienda en Euros. Es la variable objetivo a predecir.

- **Area**:  
  Área de la vivienda en metros cuadrados. Generalmente, a mayor superficie, mayor precio.

- **Room**:  
  Número de habitaciones de la vivienda. Un mayor número de habitaciones puede implicar un mayor precio.

- **Lon**:  
  Longitud de la ubicación de la vivienda. Relacionada con la localización geográfica.

- **Lat**:  
  Latitud de la ubicación de la vivienda. Similar a la longitud, influye en la ubicación y el precio.

## Tecnologías Utilizadas

- **Python**
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Streamlit**

## Análisis Exploratorio de Datos (EDA) y Preprocesamiento

Se realizó un análisis para identificar correlaciones entre las variables y detectar patrones que pudieran influir en la predicción de los precios de las viviendas.

### Preprocesamiento de los Datos:

- **Eliminación de columnas innecesarias**: Se eliminaron las columnas `Unnamed: 0`, `Address` y `Zip`, ya que no aportan información significativa.
- **Eliminación de valores nulos**: Se eliminaron los valores nulos, ya que representan solo el 0.54% del dataset.

## Modelado y Evaluación de Modelos

Se entrenaron varios modelos de regresión para predecir los precios de las viviendas:

### Modelos Entrenados:

- **Linear Regression**: Modelo base que establece una relación lineal entre las variables independientes y la variable objetivo.
- **Lasso**: Regresión lineal con penalización L1 que realiza selección de variables.
- **Stochastic Gradient Descent (SGDRegressor)**: Optimización iterativa para entrenar modelos lineales, útil para datasets grandes.
- **SVR (Support Vector Regression)**: Algoritmo de SVM adaptado a regresión, robusto frente a outliers.
- **KNeighborsRegressor**: Modelo basado en la media de las muestras más cercanas en el espacio de características.
- **DecisionTreeRegressor**: Modelo de árbol de decisión que ajusta predicciones locales.
- **RandomForestRegressor**: Ensamble de árboles de decisión que mejora la generalización y reduce el overfitting.
- **GradientBoostingRegressor**: Algoritmo de boosting secuencial que ajusta los errores de los modelos anteriores.

## Métricas Utilizadas

- **RMSE (Root Mean Squared Error)**: Mide la diferencia promedio entre los valores predichos y los valores reales. Un valor más bajo de RMSE indica un mejor modelo.
- **MSE (Mean Squared Error)**: Calcula el promedio de las diferencias al cuadrado entre los valores estimados y los valores reales.
- **MAE (Mean Absolute Error)**: Mide la diferencia promedio en términos absolutos entre las predicciones y los valores reales.
- **R² (Coeficiente de Determinación)**: Mide la calidad del ajuste del modelo. Un valor cercano a 1 indica un buen modelo.
- **Validación cruzada:** Se utilizar una técnica de validación cruzada para asegurarse de que el modelo generaliza bien a datos no vistos, minimizando el sobreajuste y garantizando que los resultados obtenidos sean robustos.

## Resultados y Conclusiones

El modelo **Gradient Boosting Regressor** fue seleccionado como el más adecuado para la predicción debido a su capacidad para captar las relaciones complejas en los datos.

### Métricas Finales:

- **RMSE**: 252166.65 €
- **MAE**: 99807.56 €
- **R²**: 0.791174

Estas métricas demuestran que el modelo tiene una capacidad predictiva aceptable y una excelente capacidad de generalización, aunque para los precios de vivienda más bajos. El R² de 0.7911 indica que el modelo explica el 79.11% de la variabilidad en los precios de las viviendas.

## Archivos del Proyecto

- **HousingPrices.csv**: Dataset utilizado para el entrenamiento del modelo.
- **cargar_datos.py**: Código para cargar los datos y dividirlos en conjuntos de entrenamiento y test.
- **data_preprocessing.py**: Código para realizar el preprocesamiento de los datos.
- **model_training.py**: Código para el entrenamiento del modelo.
- **GradientBoostingRegressor_model_precios_viviendas.pkl**: modelo entrenado.

## Ejecución Local

1. Clona el repositorio:
   git clone https://github.com/monicabernabe/Prediccion_precios_viviendas
   
2. Instala las dependencias:
  pip install -r requirements.txt

3. Ejecuta la app:
  streamlit run app.py

## 🌐 Demo en vivo

👉 [Haz clic aquí para ver la app en Streamlit]([https://tu-enlace.streamlit.app](https://monicabernabe-prediccion-precios-viviendas-app-7icjkh.streamlit.app/))

### Descripción del Proyecto

Esta aplicación permite predecir el precio de una vivienda en Ámsterdam utilizando un modelo de **Gradient Boosting Regressor**. Con solo ingresar el área de la vivienda, el número de habitaciones, y las coordenadas geográficas (latitud y longitud), la app estima el precio de la vivienda. El modelo fue entrenado con datos reales de precios de viviendas en Ámsterdam, por lo que puede ayudar a evaluar el valor de una propiedad en esa área.

## Autora
Mónica Bernabé
Ingeniera Técnica Industrial | Aseguramiento de calidad | Consultora en Validación de Sistemas | Ciencia de Datos
