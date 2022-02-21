# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 08:11:22 2022

@author: k_oso
"""

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(verbose=False, warm_start=True)

#Creamos las tres opciones del juego
options = ["piedra", "tijeras", "papel"]

# esta funsión busca quien es el ganador y da como resultado un número aludiendo al player ganador.
def search_winner(p1, p2):
    if p1 == p2:
        result = 0
        
    elif p1 == "piedra" and p2 == "tijeras":
        result = 1
    elif p1 == "piedra" and p2 == "papel":
        result = 2
    elif p1 == "tijeras" and p2 == "piedra":
        result = 2
    elif p1 == "tijeras" and p2 == "papel":
        result = 1
    elif p1 == "papel" and p2 == "piedra":
        result = 1
    elif p1 == "papel" and p2 == "tijeras":
        result = 2
        
    return result    

# se crea un test para la busqueda dle ganador lo que se hace es saber si al buscar el ganador la funsión entrega el resultado 
# correcto
test = [
    ["piedra", "piedra", 0 ],
    ["piedra", "tijeras", 1],
    ["piedra", "papel", 2]
]

for partida in test:
    print(partida)
    print("Player1 %s Player2: %s Winner: %s Validation %s" % (
        partida[0], partida[1], search_winner(partida[0], partida[1]), partida[2]
    ))
    
# para elegir de forma aleatoria las tres tipos de opciones, piedra , pepel o tijeras.
from random import choice
def get_choice():
    return choice(options)

# convertimos la opción en una matriz de 1x3 con valores de 0 y 1
def str_to_list(option):
    if option=="piedra":
        res = [1, 0, 0]
    elif option=="tijeras":
        res = [0, 1, 0]
    elif option=="papel":
        res = [0, 0, 1]
    
    return res

#asiganamos la lista a una variable
data_X = list(map(str_to_list, ["piedra", "tijeras", "papel"]))
data_y = list(map(str_to_list, ["papel", "piedra", "tijeras"]))

print(data_X) # la mostramos para corroborar.
print(data_y)

# Entrenamos al modelo, parte muy importante para entrenar el modelo.
model = clf.fit([data_X[0]], [data_y[0]])
print(model)
print(model.coefs_)

# aquí la neurona aprende a jugar 
def play_and_learn(iters=10, debug=False):
    #iniciamos en 0 los win y los loss
    score = {"win": 0, "loose": 0}
    
    data_X = [] # sin valores
    data_y = [] # sin valores
    # iters sera el número de vueltas en el ciclo for para aprender
    for i in range(iters):
        player1 = get_choice()
        # aquí predecimos la probabilidad en porcentaje de ganarle a player1
        predict = model.predict_proba([str_to_list(player1)])[0]
        # aquí seleccionamos cual es la mejor alternativa para ganar a player1 según la predicción en porcentaje de acierto
        if predict[0] >= 0.95:
            player2 = options[0]
        elif predict[1] >= 0.95:
            player2 = options[1]
        elif predict[2] >= 0.95:
            player2 = options[2]
        else:
            player2 = get_choice()
        
        if debug==True:
            print("Player1: %s  Modelo %s Player2: %s -->" % (player1, predict, player2))
        
        winner = search_winner(player1, player2)

        if debug==True:
            print("Comprobamos: p1 VS p2: %s" % winner)
            
        if winner==2:
            data_X.append(str_to_list(player1))
            data_y.append(str_to_list(player2))
            score["win"]+= 1
        else:
            score["loose"]+=1
        
    return score, data_X, data_y

score, data_X, data_y = play_and_learn(1, debug=True)
print(data_X)
print(data_y)
print("score: %s el porciento %s %%" % (score, (score["win"]*100/(score["win"]+score["loose"]))))
if len(data_X):
    model = model.partial_fit(data_X, data_y)
    
i = 0
historico_pct = []
while True:
    i+=1
    score, data_X, data_y = play_and_learn(iters=1000, debug=False)
    pct = (score["win"]*100/(score["win"]+score["loose"]))
    historico_pct.append(pct)
    print("Iter: %s - score: %s %s %%" % (i, score, pct))
    
    if len(data_X):
        model = model.partial_fit(data_X, data_y)
    
    if sum(historico_pct[-9:])==900:
        break

print('Escoge un valor: ')
preOption = input()
predict = model.predict_proba([str_to_list(preOption)])[0]
print('Predicccion: {}'.format(predict[0]))
