# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:13:49 2019

@author: Laura Rabadán Ortega
"""
import numpy as np
import cv2 
from matplotlib import pyplot as plt

# Función para leer una imagen, en color o grises, y trabajar con cv2
# ARGUMENTOS DE ENTRADA:
#   -> filename: path de la imagen que se quiere abrir
#   -> flagColor: flag que indica el color con el que se quiere leer la imagen
#                 0 -> grises     
#                 1 -> color
def LeeImagenCV (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
    img = img.astype(float)
    
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
    
    imgrgb = imgrgb.astype(float)
    
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
                    
                    new_matrix[i][j][k] = int(matrix[i][j])
                    
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

# Función para cambiar el color de una lista de píxeles 
# ARGUMENTOS DE ENTRADA:
#   -> imagen: matriz de la imnagen que se quiere normalizar
#   -> pixel: lista de píxeles que se quieren modificar
#   -> color: lista de colores que le corresponden a cada color
def CambiaPixel (imagen, pixel, color):
    
    # Se garantiza que la imagen sea tribanda y esté bien ajustada
    imagen = VisualizaMatriz(imagen)
    
    # Se recorre la lista de píxeles y se le cambia el color por el color 
    # correspondiente, ajustandolo al rango
    for i in range (0, len(pixel)):
        
        a = pixel[i][0]
        b = pixel[i][1]
        
        imagen[a][b][0] = (imagen[a][b][0]+color[i][0])%255
        imagen[a][b][1] = (imagen[a][b][1]+color[i][1])%255
        imagen[a][b][2] = (imagen[a][b][2]+color[i][2])%255
    
    # Se devuelve la imagen modificada
    return imagen 
    
# Función para visualizar varias imágenes en una sola ventana y cada una con 
# su título
# ARGUMENTOS DE ENTRADA:
#   -> vim: lista de imágenes que se quiere mostrar
#   -> x: número de filas 
#   -> y: número de columnas 
#   -> titles: lista de títulos para  poner en cada imagen
def pintaVarias (vim, x, y, titles):
    
    # Contador para el id de la imagen
    cont = 1
    
    # Se recorren todas las imagenes de la lista y se pintan
    for i in range (len(vim)):
        
        vim[i] = vim[i].astype(np.uint8)
        
        # Identificador de la imagen
        ind = str(x) + str(y) + str(cont)
        
        # Se pinta la subimagen
        ax = plt.subplot(str(ind))
        ax.set_title(titles[i])
        ax.imshow(vim[i])

        # Se aumenta el índice
        cont += 1
    
    # Se muestra las imágenes 
    plt.tight_layout()                  # Márgenes entre las imágenes
    plt.show()

##############################################################################
##############################################################################

# Ejemplos de funcionamiento

# Ejercicio 1

# Imágenes para trabajar con cv2
logo = LeeImagenCV('logoOpenCV.jpg',1)
messi = LeeImagenCV('messi.jpg',0)

# Imágenes para trabajar con matplotlib
dave = LeeImagenPLT('dave.jpg',0)
oren = LeeImagenPLT('orapple.jpg',1)

# Ejercicio 2

# Se convierte una imagen monobanda en tribanda
messi1 = messi + 255

print("\nImagen monobanda y sin normalizar\n", messi1)
print("\nValor máximo: %d \tValor mínimo: %d" % (messi1.max(), messi1.min()))

input('Pulsar \'enter\' ')

messi1 = VisualizaMatriz(messi1)

print("\nValor máximo: %d \tValor mínimo: %d" % (messi1.max(), messi1.min()))

# Ejercicio 3

input('Pulsar \'enter\' ')

print("\nVisualizar 2 imágenes a la vez")

print("\t -> CV2")

pintaMI([logo,messi], 'CV')

print("\t -> PLT")
pintaMI([dave,oren], 'PLT')

input('Pulsar \'enter\' ')


# Ejercicio 4

print("Modificar el color de una lista de píxeles")

c = [(128,0,0),(128,128,0), (0,128,128)]

pixel = []
color = []

for i in range (108, 209):
    for j in range (126, 227):
        
        pixel.append((i,j))
        color.append(c[j%3])

print("\t -> CV:")
imagen = CambiaPixel(messi, pixel, color)   
imagen = imagen.astype(np.uint8)
cv2.imshow('Cambia pixel',imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\t -> PLT:")
imagen = CambiaPixel(dave, pixel, color)  

imagen = imagen.astype(np.uint8)
plt.title("Cambia pixel")
plt.imshow(imagen)
plt.show()

input('Pulsar \'enter\' ')

# Ejercicio 5

print("Representar varias imágenes con su título")

logo = logo.astype(np.uint8)
messi = messi.astype(np.uint8)

logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
messi = cv2.cvtColor(messi, cv2.COLOR_BGR2RGB)

pintaVarias([logo, oren, dave, messi], 2, 2, ['Logo OpenCV', 'Orapple', 'Dave', 'Messi'])
