# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 00:04:52 2022

@author: k_oso
"""

#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#Importamos los datos de la misma librería de scikit-learn
diabetes = datasets.load_diabetes()
diabetes2 = pd.read_csv("diabetes.csv")

#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(diabetes.keys())


#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(diabetes.data.shape)

#Verifico la información de las columnas
print('Nombres columnas:')
print(diabetes.feature_names)

#Seleccionamos solamente la columna 2 del dataset
#X = diabetes.data[:, np.newaxis, 2]
X = np.reshape(diabetes2.iloc[:, 2].values, (-1,1))

#Defino los datos correspondientes a las etiquetas
#y = diabetes.target
y = diabetes2.iloc[:, 10].values

#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Indice de masa corporal')
plt.ylabel('Valor medio')
plt.show()

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Split the data into training/testing sets
X_train = X[:-20]
X_test = X[-20:]

# Split the targets into training/testing sets
y_train = y[:-20]
y_test = y[-20:]

#Defino el algoritmo a utilizar
lr = linear_model.LinearRegression()

#Entreno el modelo
lr.fit(X_train, y_train)

#Realizo una predicción
Y_pred = lr.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Indice de masa corporal')
plt.ylabel('Valor medio')
plt.show()

print('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)

print('Precisión del modelo:')
print(lr.score(X_train, y_train))

print('Ingresa el valor del imc: ')
imc = input()
targetPred = lr.predict([[imc]])
print(int(targetPred))