# Importar librerías
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.cargar_datos import load_and_split_data
from src.data_preprocessing import cleaning_data

# Cargar  y limpiar los datos
train_data, test_data = load_and_split_data('data/HousingPrices.csv')
train_data = cleaning_data(train_data)
test_data = cleaning_data(test_data)

#Separar variables
y_train = train_data['Price']
y_test = test_data['Price']
X_train = train_data.drop('Price', axis=1)
X_test = test_data.drop('Price', axis=1)

# Definir los hiperparametros
gb_model_2 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1
)

# Entrenar el modelo
gb_model_2.fit(X_train, y_train)

# Realizar predicciones
y_pred = gb_model_2.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")