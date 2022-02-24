# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:34:13 2022

@author: k_oso
"""

#Cargar el modelo entrenado
from joblib import load

model = load('pptJuego.joblib')
print(model)
print(model.coefs_)
print(model.n_layers_)

#Creamos las tres opciones del juego
options = ["piedra", "tijera", "papel"]

#Convertimos el string a un array
def str_to_list(option):
    if option=="piedra":
        res = [1, 0, 0]
    elif option=="tijera":
        res = [0, 1, 0]
    elif option=="papel":
        res = [0, 0, 1]
    
    return res

print('-----Juego del Piedra, Papel o Tijera-----')

x = 1

while x <= 1:
    #Pedimos opción al usuario
    print('Escribe una de estas opciones: piedra, papel o tijera')
    player1 = input()
    predict = model.predict_proba([str_to_list(player1)])[0]
    # aquí seleccionamos cual es la mejor alternativa para ganar a player1 según la predicción en porcentaje de acierto
    if predict[0] >= 0.95:
        player2 = options[0]
    elif predict[1] >= 0.95:
        player2 = options[1]
    elif predict[2] >= 0.95:
        player2 = options[2]
    
    print('Escogiste: {}'.format(player1))
    print('Las probabilidades de las opciones quedaron: {}'.format(predict))
    print('La maquina escogio: {}'.format(player2))
    
    print('¿Desea jugar otra vez? (1. Si, 2. No)')
    x = int(input())

#print("Player1: %s  Modelo %s Player2: %s -->" % (player1, predict, player2))