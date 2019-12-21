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

# Función para crear la pirámide Gaussiana de una imagen pasada como argumento
def PiramideGaussiana (imagen, sigma, incremento, borde, niveles):
    
    # La imagen original será la primera de la pirámide
    imagenes = [imagen]
    
    # Bucle en función de los niveles que se quiere en la pirámide
    for i in range (0, niveles - 1):
        
        # Se calculan las máscaras con el sigma dado
        k = cv2.getGaussianKernel(int(6*sigma + 1), sigma)
        
        # Se interpola la imagen y se calcula la convolución con la máscara
        aux = cv2.pyrDown(imagenes[i])
        imagenes.append(Convolution1D(aux, k, k.transpose(), borde))
        
        # Se incrementa el sigma
        sigma = sigma * incremento         
    
    # Se pinta la pirámide conseguida
    piramide = Piramide(imagenes)

    # Se devuelve el conjunto de imágenes y la piramide compuesta
    return imagenes, piramide

# Función para realizar la convolución de una imagen con dos máscaras 1D 
def Convolution1D (imagen, kernelf, kernelc, borde=cv2.BORDER_REFLECT_101):
    
    # Se utiliza la función filter2D para desplazar las máscaras por la imagen
    img = cv2.filter2D(imagen, -1, kernelf, borderType=borde)
    img = cv2.filter2D(img, -1, kernelc, borderType=borde)
    
    # Se devuleve la imagen transformada
    return img

# Función para componer las imágenes que forman la pirámide y poder mostrarla
# PRE: solo sirve para 4 niveles. 
def Piramide (imagenes):
    
    # El tamaño de la nueva imagen será de alto igual que la imagen original y
    # de largo la suma del largo de la primera y segunda imagen
    x = imagenes[0].shape[0] 
    y = imagenes[0].shape[1] + imagenes[1].shape[1]
    
    # Se crea la matriz de la imagen del tamaño acordado 
    if imagenes[0].ndim == 3:
        piramide = np.full((x,y,3),255)
    
    else:
        piramide = np.full((x,y),255)
    
    # Se copia la primera imagen
    for i in range (0, imagenes[0].shape[0]):
        for j in range (0, imagenes[0].shape[1]):
            
            piramide[i,j] = imagenes[0][i,j]
    
    # Se actualiza el incremento de la casilla que ocupan las siguientes imágenes
    y = imagenes[0].shape[1]
    
    # Segunda imagen
    for i in range (0, imagenes[1].shape[0]):
        for j in range (0, imagenes[1].shape[1]):
            
            piramide[i, j + y] = imagenes[1][i,j]
    
    x = imagenes[1].shape[0]
    
    # Tercera imagen
    for i in range (0, imagenes[2].shape[0]):
        for j in range (0, imagenes[2].shape[1]):
            
            piramide[i + x, j + y] = imagenes[2][i,j]
    
    x = imagenes[1].shape[0] + imagenes[2].shape[0]
    
    # Cuarta imagen
    for i in range (0, imagenes[3].shape[0]):
        for j in range (0, imagenes[3].shape[1]):
            
            piramide[i + x, j + y] = imagenes[3][i,j]
    
    # se devuleve la imagen compuesta
    return piramide


img = LeeImagenCV ('Tablero1.jpg',0)

m = cv2.cornerEigenValsAndVecs(img, blockSize=5, ksize=3)
b = m.reshape(img.shape[0], img.shape[1], 3, 2)
a = b[:,:,0]

print(m[0][0][:2])
print(a[0][0])
#aux = np.full((img.shape[0],img.shape[1],2), 0, dtype=np.float64)
#suma = np.full((img.shape[0],img.shape[1],2), 0, dtype=np.float64)
#prod = np.full((img.shape[0],img.shape[1],2), 0, dtype=np.float64)

#for i in range (0,img.shape[0]):
#    for j in range (0, img.shape[1]):
#        suma[i][j] = m[i][j][0] + m[i][j][1]
#        prod[i][j] = m[i][j][0] * m[i][j][1]
#        aux[i][j][0] = m[i][j][0]
#        aux[i][j][1] = m[i][j][1]

#a = aux[:][:][0] * aux[:][:][1] 
#b = aux[:][:][0] + aux[:][:][1]


f = m

# Se cogen las direcciones (derivadas)

# Supresión de no máximos
radio = 5
puntos = []

verificador = np.full((f.shape[0],f.shape[1]), True, dtype=np.bool)

for i in range (radio - 1, i < f.shape[0], radio):
    
    for j in range (radio - 1, j < f.shape[0][0], radio):
        
        maximo = -1
        ind = 0
        
        if verificador[i][j]:
            
            for k in range (0, radio):
                for l in range (0, radio):
                    
                    if f[i+k][j+l] > maximo:
                        maximo = f[i+k][j+l]
                        ind = [i+k, j+l]
            



# Supresión de no maximos 
# 1. Decidir a que distancia se quiere que estén los máximos. Nomás de 10 píxeles. La funciónd debe depender de ese valor. 
# 2. El punto central debe ser mayor que todos los puntos de la ventana. 
# 3. Ahorrar cálculos: 
#        Si se consigue un máximo local, los qeu comparten ventana no pueden ser máximos locales. 
#        Usar una ventana auxiliar booleana que indique si se va a estudiar el píxel o no 
#        Inicializarla a SI a todo. 
#        Se fuerza a que haya puntos peores pero dispersos
