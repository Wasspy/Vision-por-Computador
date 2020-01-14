# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:52:52 2019

@author: Laura Rabadán Ortega
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
import random

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
np.seterr(divide='ignore', invalid='ignore')

"""
    #########################################################################
    
    Funciones adicionales para trabajar con las imágenes
    
    #########################################################################    
"""

def LeeImagenCV (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
#    img = img.astype(np.float)
    
    if (img.size == 0):
        print('Error')
    
    return img

#############################################################################
    
# Función para realizar la convolución de una imagen con dos máscaras 1D 
def Convolution1D (imagen, kernelf, kernelc, borde=cv2.BORDER_REFLECT_101):
    
    # Se utiliza la función filter2D para desplazar las máscaras por la imagen
    img = cv2.filter2D(imagen, -1, kernelf, borderType=borde)
    img = cv2.filter2D(img, -1, kernelc, borderType=borde)
    
    # Se devuleve la imagen transformada
    return img

#############################################################################
    
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

#############################################################################
    
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

#############################################################################
# Función para crear la pirámide Gaussiana de una imagen pasada como argumento
def PiramideGaussiana (imagen, sigma, incremento, niveles):
    
    # La imagen original será la primera de la pirámide
    imagenes = [imagen]
    
    # Bucle en función de los niveles que se quiere en la pirámide
    for i in range (0, niveles - 1):
        
        # Se calculan las máscaras con el sigma dado
        k = cv2.getGaussianKernel(int(6*sigma + 1), sigma)
        
        # Se interpola la imagen y se calcula la convolución con la máscara
        aux = cv2.pyrDown(imagenes[i])
        imagenes.append(Convolution1D(aux, k, k.transpose(), cv2.BORDER_REFLECT_101))
        
        # Se incrementa el sigma
        sigma = sigma * incremento

    # Se devuelve el conjunto de imágenes y la piramide compuesta
    return imagenes

#############################################################################
#############################################################################

"""
    #########################################################################
    
    Práctica 3 - Detección de puntos relevantes y Construcción de panoramas
    
    #########################################################################
    #########################################################################
    
    Ejercicio 1:
    
    #########################################################################
""" 

# Función para sacar los KeyPoints de una imagen
def ConseguirKeyPoints(img, umbral, nivel, radio):
    
    num_keypoints = 0
    
    # Puntos Harris
    harris = cv2.cornerEigenValsAndVecs(img, blockSize=5, ksize=3)
    harris = harris.reshape(img.shape[0], img.shape[1], 3, 2)         # [[l1 l2][xi x2] [y1 y2]]

    lamdas = harris[:,:,0]       # Nos quedamos solo con las lamdas

    f = (lamdas[:,:,0] * lamdas[:,:,1]) / (lamdas[:,:,0] + lamdas[:,:,1])

    # Dirección de los puntos
    sigma = 4.5
    tam = int(sigma*6 + 1)

    k = cv2.getGaussianKernel(tam, sigma)  # Tamaño asociado al sigma dado, calculado con tam=(int)(sigma*6 + 1)
    
    dx = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.filter2D(img, -1, k.transpose(), borderType=cv2.BORDER_REFLECT_101)
    
    direcciones =  dy / dx
    direcciones = direcciones * (180.0 / math.pi)       # Se pasa de radianes a grados
    
    # Escala 
    escala = radio
  
    # Supresión de no máximos
    alto, ancho = f.shape[:2]
    
    puntos = np.empty((1,2), dtype=np.int)              # Puntos seleccionados
    
    verificador = np.full((alto, ancho), True, dtype=np.bool) # Ventana que indica si hay que
                                                              # estudiar o no un punto como 
                                                              # posible máximo
    for i in range (0, alto - radio + 1):
    
        for j in range (0, ancho - radio + 1):
            
            if verificador[i][j]:       # Si ese punto está permitido, se calcula
                                        # la ventana que tiene ese punto en la esquina
                                        # superior izquierda
                maximo = -1
                ind = 0
                encontrado = False
        
                # Se recorre la ventana
                for k in range (0, radio):
                
                    for l in range (0, radio):
                        
                        # Si es un valor permitido y es el mayor de la ventana y superior al umbral, se guarda
                        if verificador[i+k][j+l] and (f[i+k][j+l] > maximo) and (f[i+k][j+l] > umbral):
                            
                            maximo = f[i+k][j+l]
                            ind = [i+k, j+l]
                            encontrado = True
                            
                        # Todos los valores de la ventana se anulan, ya que no podrán ser máximos 
                        verificador[i+k][j+l] = False
                
                # En caso de que se encuentre algún punto válido, se guarda en la
                # lista de puntos y se vuelve a activar el punto máximo
                if encontrado:
                    verificador[ind[0]][ind[1]] = True
                    puntos = np.vstack((puntos,np.array(ind)))
                    num_keypoints += 1


    puntos = np.delete(puntos, 0, 0)  # Eleminamos el elemento inical que era basura
    
    keypoints = []
    
    # Creamos nuestra lista de KeyPoints
    if puntos.shape[0] > 0:
        for i in range (0, puntos.shape[0]):
            keypoints.append(cv2.KeyPoint(puntos[i,1], puntos[i,0], _size=escala, _angle=direcciones[puntos[i,0],puntos[i,1]]))
            
    else:
        print("\n No se encuentran puntos en el nivel %d con rango %d y umbral %f\n" % (nivel, radio, umbral))
    
    # Se devuelve la lista de KeyPoints y el número de puntos encontrados
    return keypoints, num_keypoints

#############################################################################
    
# Función para detectar las coordenadas subpixel de los KeyPoints conseguidos
def CornerSubPixel(img, puntos):
    
    zoom = 5
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03)
    
    # Se crean los nuevos puntos en el formato que admite la función
    coordenadas = np.empty((len(puntos),1,2), dtype=np.float32)
    
    for i in range (0, len(puntos)):
        coordenadas[i][0][0] = puntos[i].pt[0]
        coordenadas[i][0][1] = puntos[i].pt[1]
        
    
    cv2.cornerSubPix(img, coordenadas, winSize=(2,2), zeroZone=(-1,-1), criteria=criteria)
    
    # Seleccionamos los puntos con los que nos vamos a quedar
    ind = [random.randint(0, len(puntos)),random.randint(0, len(puntos)),random.randint(0, len(puntos))]
    
    nuevos_puntos = [puntos[ind[0]], puntos[ind[1]], puntos[ind[2]]]
    
    keypoint = cv2.KeyPoint(coordenadas[ind[0]][0][0], coordenadas[ind[0]][0][1], _size=puntos[ind[0]].size, _angle=puntos[ind[0]].angle)
    nuevas_coordenadas = [keypoint]
    
    keypoint = cv2.KeyPoint(coordenadas[ind[1]][0][0], coordenadas[ind[1]][0][1], _size=puntos[ind[1]].size, _angle=puntos[ind[1]].angle)
    nuevas_coordenadas.append(keypoint)
    
    keypoint = cv2.KeyPoint(coordenadas[ind[2]][0][0], coordenadas[ind[2]][0][1], _size=puntos[ind[2]].size, _angle=puntos[ind[2]].angle)
    nuevas_coordenadas.append(keypoint)
    
    # Pintamos las coordenadas en una imagen 10x10 con un zoom 5x
    for i in range (0, 3):
        
        x = int(nuevos_puntos[i].pt[1])
        y = int(nuevos_puntos[i].pt[0])
        
        x = [x-5, x+5]
        y = [y-5, y+5]
        
        if x[1] < 0:
            x[0] -= x[1]
            x[1] = 0
        
        if x[0] > img.shape[0]:
            x[1] -= (x[0] - img.shape[0])
            x[0] = img.shape[0]
            
        if y[0] < 0: 
            y[1] -= y[0]
            y[0] = 0
        
        if y[1] > img.shape[1]:
            y[0] -= (y[1] - img.shape[0])
            y[1] = img.shape[1]
            
        
        original = [nuevos_puntos[i].pt[1] - x[0], nuevos_puntos[i].pt[0] - y[0]]
        corregida = [nuevas_coordenadas[i].pt[1] - x[0], nuevas_coordenadas[i].pt[0] - y[0]]
        
        imagen = img[x[0]:x[1],y[0]:y[1]]
        
        alto, largo = imagen.shape
        
        imagen = cv2.resize(imagen, (alto*zoom, largo*zoom))
        
        imagen = AjustaTribanda(imagen)
        imagen = NormalizarIntervalo(imagen)
    
        # Se pintan las imágenes
        plt.title("CornerSubPixel - " + str(i))
        
        plt.imshow(imagen)
        
        plt.scatter(original[0], original[1], s=nuevos_puntos[i].size*5, facecolors='none', edgecolors='r', label="Original")
        plt.scatter(corregida[0],original[1], s=nuevas_coordenadas[i].size*5, facecolors='none', edgecolors='b', label="Corregida")
        
        plt.legend()
        plt.show()
        
        input("\n ENTER para mostrar la siguiente imagen: ")
 

#############################################################################
#############################################################################
  
# Ejercicio 1
def Ejercicio1 (nombre_imagen, umbral, sigma, incremento, niveles, radio):
    
    # Se leen las imagen
    imagen = LeeImagenCV(nombre_imagen,0)
     
    # Se calculan los niveles de la pirámide Gaussiana
    imagenes = PiramideGaussiana (imagen, sigma, incremento, niveles)
    
    total_keypoints = 0
    
    puntos_por_nivel = []
    
    # Se recorren todos los niveles de la pirámide y se calculan sus KeyPoints
    for i in range (0, niveles):
        
        print("\n Calculando puntos del nivel %d ... " % (i + 1))
    
        keypoints, num_keypoints = ConseguirKeyPoints(imagenes[i], umbral[i], i, radio[i])
        
        print(" -> Puntos conseguidos: ", num_keypoints)
        if num_keypoints > 0:
            puntos_por_nivel.append(keypoints)
        
            total_keypoints += num_keypoints
           
    print("\n Umbrales: " + str(umbral))
    print("\n Puntos: ", total_keypoints)
    
    # Se pinta cada nivel de la pirámide con los puntos conseguidos en ella
    for i in range (0, len(imagenes)):
        
        img = AjustaTribanda(imagenes[i])
        img = NormalizarIntervalo(img)
    
        puntos = puntos_por_nivel[i]
        
        # Se pintan las imágenes
        plt.title("Nivel " + str(i))
        
        imagen_puntos = cv2.drawKeypoints(img, puntos, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.imshow(imagen_puntos)
        
        plt.show()
        
        input("\n ENTER para mostrar la siguiente imagen: ")
    
    print("\n -> CornerSubPixel")
    
    # CornerSubPixel
    CornerSubPixel(imagenes[0],puntos_por_nivel[0])

"""
    #########################################################################
    
    Ejercicio 2:
        
    #########################################################################
""" 
# Función para conseguir los puntos y descriptores Akaze
def Akaze (imagen1, imagen2):
    
    akaze = cv2.AKAZE_create()
    
    keypoints1, descriptor1 = akaze.detectAndCompute(imagen1, None)
    keypoints2, descriptor2 = akaze.detectAndCompute(imagen2, None)
    
    return keypoints1, descriptor1, keypoints2, descriptor2

#############################################################################
    
# Función para establecer correspondencia entre puntos de las dos imágenes
# crossCheck indica si se quiere que sea Fuerza Bruta o Knn
def Correspondecia (descriptor1, descriptor2, crossCheck):
    
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck)
    
    if crossCheck:
        
        return matches.match(descriptor1, descriptor2)
    
    return matches.knnMatch(descriptor1, descriptor2, k=2) 

#############################################################################
    
# Función para poder los matches conseguidos por medio de Lowe's Ratio Distance
def LoweRatio (matches, umbral):
    
    puntos = []
    
    for k in matches:
        
        # Se utiliza la fórmula: (d1/d2) < umbral => d1 < umbral * d2
        if k[0].distance < umbral * k[1].distance:
            puntos.append(k[0])
    
    return puntos

#############################################################################
    
# Función para pintar las correspondencias conseguidas
def PintarMatches (imagen1, imagen2, kp1, kp2, matches, titulo):

    # Flag = 2 -> NOT_DRAW_SINGLE_POINTS
    imagen = cv2.drawMatches(imagen1, kp1, imagen2, kp2, matches, outImg=np.array([]), flags=2) 
    
    img = AjustaTribanda(imagen)
        
    img = NormalizarIntervalo(img)
    
    img = img.astype(np.uint8)
    
    cv2.imshow(titulo, img)
 

#############################################################################
#############################################################################
  
# Ejercicio 2
def Ejercicio2 (nombres, umbral):
    
    # Se leen las imagen
    imagen1 = LeeImagenCV(nombres[0],1)
    imagen2 = LeeImagenCV(nombres[1],1)
    
    # Puntos y descriptores Akaze
    keypoints1, descriptor1, keypoints2, descriptor2 = Akaze(imagen1, imagen2)
    
    
    matchesFB = Correspondecia (descriptor1, descriptor2, True)
    matchesKnn = Correspondecia (descriptor1, descriptor2, False)
    
    # Se mezclan los puntos conseguidos para coger los 100 primeros de forma aleatoria
    np.random.shuffle(matchesFB)
    np.random.shuffle(matchesKnn)
    
    # Puntos seleccionados por medio de Lowe's ratio distance
    puntos = LoweRatio (matchesKnn, umbral)
    
    print("\n -> Numeros de matches con Fuerza Bruta: ", len(matchesFB))
    print(" -> Numero de matches con Lowe-Arange-2NN: ", len(puntos))
     
    # Se pintan los resultados conseguidos
    PintarMatches (imagen1, imagen2, keypoints1, keypoints2, matchesFB[:100], "FuerzaBruta - 100")
    PintarMatches (imagen1, imagen2, keypoints1, keypoints2, puntos[:100], "Lowe-Average-2NN - 100")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#############################################################################
#############################################################################
    
"""
    Programa principal
"""    

def main():
    
    # Ejercicio 1
    nombres_imagen = ["imagenes/Yosemite1.jpg","imagenes/Yosemite2.jpg"]
    
    sigma = 0.1
    incremento = 1.2
    niveles = 4
    radio = [20,10,10,10]
    umbral = [0.001,0.00045, 0.0001,0.001]
    
    print ("\n EJERCICIO 1: Deteccion de puntos HARRIS \n -> Yosemite1.jpg")
    Ejercicio1 (nombres_imagen[0], umbral, sigma, incremento, niveles, radio)
    
    print ("\n ->Yosemite2.jpg")
    Ejercicio1 (nombres_imagen[1], umbral, sigma, incremento, niveles, radio)
    
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    
    # Ejercicio 2
    nombres_imagen = ["imagenes/Yosemite3.jpg","imagenes/Yosemite4.jpg"]
    
    umbral = [0.4, 0.7]
    
    print ("\n EJERCICIO 2: Detección y extracción de descriptores AKAZe ")
    
    print (" -> Ficheros: %s - %s" % (nombres_imagen[0], nombres_imagen[1]))
    print ("\n - Usando un umbral de ", umbral[0])
    Ejercicio2 (nombres_imagen, umbral[0])
    
    print ("\n - Usando un umbral de ", umbral[1])
    Ejercicio2 (nombres_imagen, umbral[1])
    
    nombres_imagen = ["imagenes/Tablero1.jpg","imagenes/Tablero2.jpg"]
    
    print ("\n -> Ficheros: %s - %s" % (nombres_imagen[0], nombres_imagen[1]))
    print ("\n - Usando un umbral de ", umbral[0])
    Ejercicio2 (nombres_imagen, umbral[0])
    
    print ("\n - Usando un umbral de ", umbral[1])
    Ejercicio2 (nombres_imagen, umbral[1])

#############################################################################
#############################################################################
 
# Se llama al programa principal para que ejecute el programa
if __name__ == '__main__':
    main()
