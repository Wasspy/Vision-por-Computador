# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:56:18 2019

@author: Laura
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2 

##############################################################################
##############################################################################
#                FUNCIONES BÁSICAS DE LA PRÁCTICA 0                          #
# Función para leer una imagen, en color o grises, y trabajar con cv2
# ARGUMENTOS DE ENTRADA:
#   -> filename: path de la imagen que se quiere abrir
#   -> flagColor: flag que indica el color con el que se quiere leer la imagen
#                 0 -> grises     
#                 1 -> color
def LeeImagenCV (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
    img = img.astype(np.float)
    
    return img

# Función para leer una imagen, en color o grises, y trabajar con matplotlib
# ARGUMENTOS DE ENTRADA:
#   -> filename: path de la imagen que se quiere abrir
#   -> flagColor: flag que indica el color con el que se quiere leer la imagen
#                 0 -> grises     
#                 1 -> color
def LeeImagenPLT (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
    # cv2trabaja con BGR en vez de RGB, por lo que si se va a trabajar con RGB 
    # hay que cambiarle el canal
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    imgrgb = imgrgb.astype(np.float)
    
    return imgrgb

# Función para ajustar una imagen monobanda a tribanda
# ARGUMENTOS DE ENTRADA:
#   -> marix: matriz de la imagen que se quiere convertir
def AjustaTribanda (matrix):
    
    # Solo se ajusta si es monobanda, si es tribanda se devuelve sin cambio
    if len(matrix.shape) == 2:
        
        # Dimensiones de la nueva matriz
        a = (matrix.shape)[0]
        b = (matrix.shape)[1]
        c = 3
        
        # Se crea la nueva matriz inicializada a 0
        new_matrix = np.zeros((a,b,c), dtype=matrix.dtype)

        # Se triplica el valor que tenía cada píxel para pasarlo a tribanda
        for i in range (0, a):
            for j in range (0, b):
                for k in range (0, c):
                    
                     new_matrix[i][j][k] = matrix[i][j].astype(int)
                    
        return new_matrix
    
    else:
        return matrix

# Función para normalizar los valores de la matriz de datos dentro del intervalo
# [0,255]
# ARGUMENTOS DE ENTRADA:
#   -> matrix: matriz de la imnagen qeu se quiere normalizar
def NormalizarIntervalo (matrix):
    
    # Solo se normaliza si la matriz tiene valores fuera del rango [0,255]
    if (matrix.max() > 255 or matrix.min() < 0):
        # Se guardan los valores máximos y mínimos de la matriz, y su diferencia
        maxi = matrix.max()
        mini = matrix.min()
        dif = maxi - mini
        
        for i in range (0, (matrix.shape)[0]):
            for j in range (0, (matrix.shape)[1]):
                for k in range (0, (matrix.shape)[2]):
                    
                    # Se normaliza cada valor de la imagen según la fórmula
                    # (normalizar en el intervalo [0,1] y ajustarlo al 
                    # correspondiente intervalo [0,255])
                    matrix[i][j][k] = (255 * (matrix[i][j][k] - mini)) // dif
                        
    return matrix

# Función para visualizar parcialmente una imagen después de modificarla de 
# monobanda a tribanda y normalizar sus valores 
# ARGUMENTOS DE ENTRADA:
#   -> matrix: matriz de la imnagen qeu se quiere normalizar
def VisualizaMatriz (matrix):
    
    # Se ajustan las imánenes monobanda a tribanda
    matrix = AjustaTribanda(matrix)
    
    # Normalizar la matriz dentro del intervalo [0,255]
    matrix = NormalizarIntervalo (matrix)
    
    # Se pinta la matriz
    print ("\nImagen ajustada y normalizada \n", matrix)
    
    return matrix

# Función para visualizar varias imágenes en una sola ventana
# ARGUMENTOS DE ENTRADA:
#   -> matrix: matriz de la imnagen que se quiere normalizar
#   -> pinta: indica en que formato se quiere visualizar. por defecto, se 
#             utiliza cv2, pero puede pintarse con matplotlib si se indica 'PLT'.
# PRE: se considera que si se quiere pintar en un formato determinado, el usuario
#      mandará las matrices con un código de color correco (RGB para matplotlib
#      y BGR para cv2)
def pintaMI (vim, pinta='CV'):
    
    # Si se pinta con cv2, las imágenes se escalan a la imagen más pequeña, ya
    # que si se escala a la mayor cv2 muestra lo que saca por pantalla. 
    if (pinta == 'CV'):
        
        # Se busca la imagen más pequeña para escalar el resto a ella
        max_size = 999999999999
            
        for i in range (0, len(vim)):
            
            # Se aprovecha para comprobar que todas las imágenes sean tribanda
            vim[i] = AjustaTribanda(vim[i])
            
            if (vim[i].size < max_size):
                max_size = vim[i].size
                ind = i
    
    # Si se pinta con matplotlib, se ajusta a la más grande, ya que si se 
    # muestran enteras y las imágenes se desforman menos
    else:
        
        # Se busca la imagen mayor para escalar el resto a ella
        max_size = 0
            
        for i in range (0, len(vim)):
            
            # Igual que en el caso anterior, se aprovecha para comprobar que 
            # todas las imágenes sean tribanda
            vim[i] = AjustaTribanda(vim[i])
            
            if (vim[i].size > max_size):
                max_size = vim[i].size
                ind = i
    
    # En ambos casos, se guardan las dimensiones de la imagen seleccionada
    dim = (vim[ind].shape[0],vim[ind].shape[1])
    
    # Se escalan todas las imagenes menos la imagen de referencia
    for i in range (0,ind):
        
        # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        # INTER_AREA: para que se base en los píxeles del area. 
        vim[i] = cv2.resize(vim[i], dim, interpolation = cv2.INTER_AREA)
        
    for i in range (ind, len(vim)):
        
        vim[i] = cv2.resize(vim[i], dim, interpolation = cv2.INTER_AREA)
    
    # Se concatenan todas las imágenes de forma vertical
    imagen = np.concatenate((vim[0],vim[1]), axis=1)
   
    for i in range (2, len(vim)):
       
       imagen = np.concatenate((imagen,vim[i]), axis=1)
   
    imagen = imagen.astype(np.uint8)
    
    # Se pinta la imagen resultante en función del parámetro 'pinta'
    if (pinta == 'CV'):
        
#        cv2.imwrite('imagen.jpg',imagen)
        cv2.imshow("Conjunto", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        plt.title("Conjunto")
        plt.imshow(imagen)
        plt.show()
        

# Reflejar los bordes. 
def Borde1 (imagen, borde):
    
    filas = imagen.shape[0] + borde*2
    columnas = imagen.shape[1] + borde*2
    
    img = np.empty((filas,columnas))
    
    # Primero se copia la imagen original en la nueva imagen
    for i in range (0, imagen.shape[0]):
        for j in range (0,imagen.shape[1]):
            
            img[i + borde, j + borde] = imagen[i,j]
    
    # Se le añade un borde en espejo por filas        
    for i in range (0, imagen.shape[0]):
        for j in range (0, borde):
            
            img[i + borde, j] = imagen[i, borde - j - 1]
            img[i + borde, -j] = imagen[i, -(borde - j + 1)]
            
    # Se le añade un borde en espejo por columnas
    for j in range (0, img.shape[1]):    
        for i in range (0, borde):    
            
            img[i, j] = img[2*borde - i - 1, j]
            img[-i,j] = img[-2*borde + i - 1, j]
            
    pintaMI([imagen, img], ["Original", "Bordes"])

    # Se devuelve la imagen con su borde
    return img

def Convolution1D (imagen, mascara):
    
    # Se crea una matriz para ahorrar tiempo de cómputo. Esta matriz tendrá 
    # el número de filas igual al número de elementos hay en el fragmento de 
    # imagen, y el mismo número de columnas que elementos de la máscara
    filas = imagen.shape[0]
    colum = mascara.shape[0]
    
    # Se calcula el número de elementos que forman el borde extra de la imagen
    bordes = (colum//2) * 2
    
    # Se crea una matriz vacia con las dimensiones anteriores
    matriz = np.empty((filas-bordes,colum))
    
    # Con ayuda de un bucle, se rellena la matriz, de manera que, si el vector 
    # original de la imagen era [1 2 3 4 5], ahora se tendría:
    # [[x 1 2] [1 2 3] [2 3 4] [3 4 5] [4 5 x]] , si la máscara es de 3. x sería
    # el borde de la imagen
    for i in range (0, matriz.shape[1] + 1):
        
        matriz[i] = imagen[i:i+colum].copy()
    
    # Se calcula la convulción de la fila de la imagen y la máscara
    return matriz @ mascara

def Convolution (imagen, filas, columnas):
    
    borde = filas.shape[0] // 2
    img = Borde1(imagen, borde)
    
    nueva = img.copy()
    
    print(imagen.shape)
    print(img.shape)
    print(nueva.shape)
    
    for i in range (0,imagen.shape[0]):
        
        aux = Convolution1D(img[i + borde], filas) 
        
        for j in range (0, imagen.shape[1]):
            nueva[i + borde,j + borde] = aux[j].copy()
    
    for i in range (0, imagen.shape[1]):
        
        aux = Convolution1D(nueva[:,i+borde],columnas)
        
        for j in range (0, imagen.shape[0]):
            img[j + borde,i + borde] = aux[j].copy()
    
#    for i in range (0, img.shape[0]):
#        for j in range (0, img.shape[1]):
#            img[i,j] += 20
#            
    print (img.max())
    print (img.min())
    nueva = AjustaTribanda(img)
    nueva = NormalizarIntervalo(nueva)
#    print (nueva.max())
#    print (nueva.min())
    
    nueva = nueva.astype(np.uint8)
    img = img.astype(np.uint8)
    nueva = cv2.cvtColor(nueva, cv2.COLOR_BGR2RGB) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    plt.imshow(nueva)
#    
    return nueva

gato = LeeImagenCV("imagenes/cat.bmp", 0) 

tam = 15

filas, columnas = cv2.getDerivKernels(1,2,tam)
sigma = tam/6

filas *= sigma
columnas *= sigma

filas = filas.reshape(filas.shape[0],)
columnas = columnas.reshape(columnas.shape[0],)

img = Borde1(gato, tam//2)

Convolution (gato, filas, columnas)

