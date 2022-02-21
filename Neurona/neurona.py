# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 00:30:48 2022

@author: k_oso
"""

import numpy as np

#Cargar lista
x1 = [0,0,0,0,1,1,1,1]
x2 = [0,0,1,1,0,0,1,1]
x3 = [0,1,0,1,0,1,0,1]
y = [0,0,0,0,0,0,0,1]
categorias = [-1,-1,-1,-1,-1,-1,-1,1]
pred = []

#Generar pesos aleatorios
weights = []
for x in range(4):
    #weights.append(random.r(-10, 10))
    weights.append(np.random.uniform(-10, 10))
print('Pesos iniciales: {}'.format(weights))

#Iterar
x, epoch = 0, 0
while x < 7:
    print('--------------Epoca {}-------'.format(epoch))
    for i in range(8):
        ec = (weights[0]*x1[i]) + (weights[1]*x2[i]) + (weights[2]*x3[i]) + weights[3]
        clasePred = 1 if ec > 0 else -1
        print('Salida Esperada: {} Salida Predecida: {}'.format(categorias[i], clasePred))
        if(clasePred != categorias[i]):
            weights[0] = weights[0] + (categorias[i]*x1[i])
            weights[1] = weights[2] + (categorias[i]*x2[i])
            weights[2] = weights[2] + (categorias[i]*x3[i])
            weights[3] = weights[3] + (categorias[i])
            #print('AJUSTANDO PESOS: {}'.format(weights))
            x = 0
        else:
            x += 1
    epoch+=1
    print(x)
    print('--------------')
   
print('Pesos Fianles {}'.format(weights))
print('Ingresa el valor de x1: ')
x1_Pred = float(input())
print('Ingresa el valor de x2: ')
x2_Pred = float(input())
print('Ingresa el valor de x3: ')
x3_Pred = float(input())
ec = (weights[0]*x1_Pred) + (weights[1]*x2_Pred) + (weights[2]*x3_Pred) + weights[3]
clasePred = 1 if ec > 0 else 0
print('Prediccion: {}'.format(clasePred))
            
