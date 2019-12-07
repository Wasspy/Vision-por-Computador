# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:13:29 2019

@author: Laura
"""

# Convolución 2D
def Convolution2D (imagen, mascara):
    
    colum = imagen.shape[1]
    borde = mascara.shape[1]//2
    
    resultado = np.empty((colum - (borde * 2),))
    
    
    for k in range (borde, colum - borde):
        
        aux = 0
        
        for i in range (0, 3): 
            for j in range (k - borde, k + borde + 1):
                
                aux += imagen[i,j]*mascara[i-(k - borde),j-(k - borde)]
        
        resultado[k - borde] = aux
    
    return resultado