# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:28 2019

@author: Laura
"""
import cv2
import numpy as np

# getDerivKernerl(dx, dy, k) -> 2 vectores, uno para fila y otro para columnas. LAS FILAS Y COLUMNAS SON LO CONRARIO (x -> columnas ; y -> filas)
# No vamos a pasar de la segunda derivada. 
# Es una derivada 2D
# k tamaño de la loseta 


#c, d = cv2.GaussianBlur()

x, y = cv2.getDerivKernels(1,0,9)

# a y b son máscaras
#   Tiene valores negativos
#   Si todos son positivos, es de alizamiento
#   Hay que normalizar la derivada
#       Se tiene que normalizar para que sume 1
# Lo único que hay que implementar es una función que convolucione un vector 
# con otro vector un vector convolucionado con otro vector. 
# Algún criterio para ampliar la imagen y que quede completa
# Se puede hacer una función de convolución 1D

# Multiplicaciçon de una matriz por un vector 
# Movimiento de memoria no cuenta
# Producto de matrices vectoriales. Evitar ciclos. 
def Convolution1D (a, b):
    
    filas = a.shape[0]
    colum = b.shape[0]
     
    bordes = (colum//2) * 2
    
    matriz = np.empty((filas-bordes,colum))
    
    for i in range (0, matriz.shape[1] + 1):
        
        matriz[i] = a[i:i+colum].copy()
    
    print (matriz)
    return matriz.dot(b)



a = np.array([(0,1,2,3,4,0), (0,5,6,7,8,0)])
b = np.array([0.3333,0.666667,0.333334])
print(a.shape)

print ("a:\n",a[0])
print ("b: \n", b)
c = Convolution1D (a[0],b)

print(c)