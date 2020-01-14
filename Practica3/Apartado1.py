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
        
        if matrix.ndim ==3: 
            for i in range (0, (matrix.shape)[0]):
                for j in range (0, (matrix.shape)[1]):
                    for k in range (0, (matrix.shape)[2]):
                        
                        # Se normaliza cada valor de la imagen según la fórmula
                        # (normalizar en el intervalo [0,1] y ajustarlo al 
                        # correspondiente intervalo [0,255])
                        matrix[i][j][k] = (255 * (matrix[i][j][k] - mini)) // dif
        
        else:
           for i in range (0, (matrix.shape)[0]):
                for j in range (0, (matrix.shape)[1]):
                        
                    # Se normaliza cada valor de la imagen según la fórmula
                    # (normalizar en el intervalo [0,1] y ajustarlo al 
                    # correspondiente intervalo [0,255])
                    matrix[i][j] = (255 * (matrix[i][j] - mini)) // dif 
                            
    return matrix

#def ConseguirKeypoints (imagen, ):
    
    

img = LeeImagenCV ("Tablero1.jpg",0)

m = cv2.cornerEigenValsAndVecs(img, blockSize=5, ksize=3)
m = m.reshape(img.shape[0], img.shape[1], 3, 2)         # [[l1 l2][xi x2] [y1 y2]]

lamdas = m[:,:,0]       # Nos quedamos solo con las lambas

f = (lamdas[:,:,0] * lamdas[:,:,1]) / (lamdas[:,:,0] + lamdas[:,:,1])


# Se cogen las direcciones (derivadas)

# Supresión de no máximos
radio = 25
puntos = np.empty((1,2), dtype=np.int)
valores = np.empty((1), dtype=np.int)

alto, ancho = f.shape[:2]

verificador = np.full((alto, ancho), True, dtype=np.bool)

for i in range (0, alto - radio + 1):
    
    for j in range (0, ancho - radio + 1):
        
        if verificador[i][j]:
        
            maximo = -1
            ind = 0
            encontrado = False
        
            for k in range (0, radio):
                
                for l in range (0, radio):
                        
                    if verificador[i+k][j+l] and (f[i+k][j+l] > maximo):
                        maximo = f[i+k][j+l]
                        ind = [i+k, j+l]
                        encontrado = True
                    
                    verificador[i+k][j+l] = False
            
            if encontrado:
                puntos = np.vstack((puntos,np.array(ind)))
                valores = np.vstack((valores, f[ind[0], ind[1]]))


puntos = np.delete(puntos, 0, 0)  # Eleminamos el elemento inical que era basura
# Orientaciones
sigma = 4.5

k = cv2.getGaussianKernel(28, 4.5)  # Tamaño asociado al sigma dado, calculado con tam=(int)(sigma*6 + 1)

dx = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
dy = cv2.filter2D(img, -1, k.transpose(), borderType=cv2.BORDER_REFLECT_101)

img_orientaciones =  dy / dx

# Escala 
escala = radio*1

keypoints = np.array(cv2.KeyPoint(puntos[0,0], puntos[0,1], _size=escala, _angle=img_orientaciones[puntos[0,0],puntos[0,1]]))

for i in range (1, puntos.shape[0]):
    
    np.vstack((keypoints,cv2.KeyPoint(puntos[i,0], puntos[i,1], _size=escala, _angle=img_orientaciones[puntos[i,0],puntos[i,1]])))


img = AjustaTribanda(img)
img = NormalizarIntervalo(img)

#outImage = []

#img2 = cv2.merge([img, img, img])
outImage = cv2.drawKeypoints(img, keypoints, img)

print("llego aqui")
# Se muestra la imagen original
plt.imshow(img)

# se pintan las regiones
plt.scatter(puntos[:,1], puntos[:,0], s=radio/2, facecolors='none', edgecolors='r', label="rango " + str(radio))


plt.show()            
#            for k in range (ind[0], radio):
#                for l in range (ind[1], radio):

# Supresión de no maximos 
# 1. Decidir a que distancia se quiere que estén los máximos. Nomás de 10 píxeles. La funciónd debe depender de ese valor. 
# 2. El punto central debe ser mayor que todos los puntos de la ventana. 
# 3. Ahorrar cálculos: 
#        Si se consigue un máximo local, los qeu comparten ventana no pueden ser máximos locales. 
#        Usar una ventana auxiliar booleana que indique si se va a estudiar el píxel o no 
#        Inicializarla a SI a todo. 
#        Se fuerza a que haya puntos peores pero dispersos
