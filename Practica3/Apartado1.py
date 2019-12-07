# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:52:52 2019

@author: Laura
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
np.seterr(divide='ignore', invalid='ignore')

def LeeImagenCV (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
#    img = img.astype(np.float)
    
    if (img.size == 0):
        print('Error')
    
    return img

img = LeeImagenCV ('Tablero1.jpg',0)

m = cv2.cornerEigenValsAndVecs(img, blockSize=5, ksize=3)

print (m[:][:][1].shape)
a = m[:][:][0] * m[:][:][1] 
b = m[:][:][0] + m[:][:][1]

f = a / b

# Supresión de no maximos 
# 1. Decidir a que distancia se quiere que estén los máximos. Nomás de 10 píxeles. La funciónd ebe depender de ese valor. 
# 2. El punto central debe ser mayor que todos los puntos de la ventana. 
# 3. Ahorrar cálculos: 
#        Si se consigue un máximo local, los qeu comparten ventana no pueden ser máximos locales. 
#        Usar una ventana auxiliar booleana que indique si se va a estudiar el píxel o no 
#        Inicializarla a SI a todo. 
#        Se fuerza a que haya puntos peores pero dispersos
