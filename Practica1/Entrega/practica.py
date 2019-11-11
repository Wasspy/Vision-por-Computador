# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:28 2019

@author: Laura
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2

##############################################################################
####################  APARTADOS Y FUNCIONES RELACIONADAS  ####################
##############################################################################

# EJERCICIO 1 - APARTADO A

# Función para realizar la convolución de una imagen con dos máscaras 1D 
def Convolution1D (imagen, kernelf, kernelc, borde=cv2.BORDER_REFLECT_101):
    
    # Se utiliza la función filter2D para desplazar las máscaras por la imagen
    img = cv2.filter2D(imagen, -1, kernelf, borderType=borde)
    img = cv2.filter2D(img, -1, kernelc, borderType=borde)
    
    # Se devuleve la imagen transformada
    return img

# Ejemplo de funcionamiento de la función Convolution1D usando distintos parámetros
def Apartado1A(gris, color, tam, sig, bor):
    
    print ("    Convolución de una imagen con máscara 1D: ")
    
    grises = []
    colores = []
    
    # Se ejecuta la función con las distintas combinaciones posibles
    for t in tam:
        for s in sig:
            for b in bor:
                k = cv2.getGaussianKernel(t, s)
                
                grises.append(Convolution1D (gris, k, k.transpose(), b))
                colores.append(Convolution1D (color, k, k.transpose(), b))
    
    # Se compara la función creada con los resultados obtenidos con la función
    # GaussianBlur
    print ("    Comparación de la función creada con GaussianBlur:")
    
    aux = cv2.GaussianBlur(gris, (31,31), 5.17, borderType=cv2.BORDER_REFLECT)
    
    PintaMI([grises[-2], aux], "Funcion propia - GassianBlur")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Se pintan los resultados obtenidos con Convolution1D
    print("     -> Parámetros usados (las máscaras son combinaciones de ellos): ")
    print("         * Tamaños: {3, 31} \n         * Sigma: {0.5, 5.17}")
    print("         * Contorno (bordes): {Espejo, Negro}")
    
    # Imágenes en escala de grises
    print("\n Imágenes en escala de grises:")
    PintaVarias([gris], 1, 1, ['Imagen Original'])
    
    PintaMI(grises[:4], "Combinaciones con tamanio de mascara de 3")
    PintaMI(grises[4:], "Combinaciones con tamanio de mascara de 31")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Imagenes a color
    print("\n Imágenes a color:")
    PintaVarias([color], 1, 1, ['Imagen Original'])
    
    PintaMI(colores[:4], "Combinaciones con tamanio de mascara de 3")
    PintaMI(colores[4:], "Combinaciones con tamanio de mascara de 31")
    
    # Pausa para mostrar las imagenes
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
# EJERCICIO 1 - APARTADO B

# Función para realizar la convolución con una máscara Laplaciana-de-Gaussiana
def Laplacian (imagen, tam, borde=cv2.BORDER_REFLECT_101):
    
    # Se calculan las máscaras qeu se van a usar, las 2 derivadas de la filas 
    # por un lado y para las columnas por otro
    kx1,ky1 = cv2.getDerivKernels(2,0,tam) 
    kx2,ky2 = cv2.getDerivKernels(0,2,tam)
    
    # Se aplican las máscaras con ayuda de la función Convolution1D
    aux = Convolution1D (imagen, kx1, ky1.transpose(), borde)
    
    # Se suman los resultados
    aux = aux + Convolution1D (imagen, kx2, ky2.transpose(), borde)
    
    # Se normalizando multiplicando la imagen resultante por sigma
    aux = aux * (((tam - 1)/6)**2)
    
    return aux

# Ejemplo de funcionamiento de la función Laplacian usando distintos parámetros  
def Apartado1B(gris, color, tam, bor):
    
    print ("    Convolución de una imagen con máscara Laplaciana: ")
    
    grises = []
    colores = []
    
    # Se prueba la funcion utilizando todas las combinaciones de parámetros
    for b in bor:
        for t in tam:
            
            grises.append(Laplacian(gris, t, b))
            colores.append(Laplacian(color, t, b))
    
    # Se compara la función creada con el resultado de la función propia de 
    # OpenCV       
    print ("\n    Comparación de la función creada con Laplacian:")
    
    aux = cv2.Laplacian(gris, cv2.CV_64F, ksize=13, borderType=cv2.BORDER_REFLECT_101)
    
    PintaMI([grises[1], aux], "Funcion propia - Laplacian")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Se muestran los resultados obtenidos con los parámetros anteriores
    print("     -> Parámetros usados (las máscaras son combinaciones de ellos): ")
    print("         * Sigma: {1, 2, 3}\n         * Contorno (bordes): {Espejo, Negro}")
    
    # Imágenes en escala de grises
    print("\n Imágenes en escala de grises:")
    PintaVarias([gris], 1, 1, ['Imagen Original'])
    
    PintaMI(grises[:3], 'Borde reflejado')
    PintaMI(grises[3:], 'Borde negro')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Imágenes a color
    print("\n Imágenes a color:")
    PintaVarias([color], 1, 1, ['Imagen Original'])
    
    PintaMI(colores[:3], 'Borde reflejado')
    PintaMI(colores[3:], 'Borde negro')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
# EJERCICIO 2 - APARTADO A

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

# Ejemplo de funcionamiento de la pirámide Gaussiana utilizando distintos parámetros. 
def Apartado2A(gris, color, sigma, incremento, niveles, bordes):
    
    print ("    Pirámide Gaussiana: ")
    
    pgrises = []
    pcolores = []
    
    # Se llama a la función con cada combinación de parámetros y se guardan los 
    # resultados conseguidos
    for b in bordes:
        grises, piragris = PiramideGaussiana(gris, sigma, incremento, b, niveles)
        colores, piracol = PiramideGaussiana(color, sigma, incremento, b, niveles)
    
        pgrises.append(piragris)
        pcolores.append(piracol)
    
    # Se pintan los resultados conseguidos
    print("     -> Parámetros usados : ")
    print("         * Sigma inicial: 0.1\n         * Incremento de sigma:1.2")
    print("         * Bordes: {Reflejado, Negro, Valor replicado}")
    
    titulo = ["Borde Reflejado", "Borde Negro", "Borde Valor Replicado"]
    
    # Se pintan las pirámides Gaussianas
    # Grises
    for i in range (0, len(pgrises)):
        
        PintaVarias([pgrises[i]], 1, 1, [str(titulo[i] + " - gris")])
    
    input ("\n\t -> Pulsar ENTER para continuar con las imágenes a color: ")
    
    for i in range (0, len(pgrises)):
        PintaVarias([pcolores[i]], 1, 1, [str(titulo[i] + " - color")])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
# EJERCICIO 2 - APARTADO B

# Función para ocnstruir la pirámide Laplaciana de una imagen pasada por argumento
def PiramideLaplaciana (imagen, sigma, incremento, borde, niveles):
        
    imagenes = []
    
    # Primero se calcula la pirámide gaussiana de la imagencon el mismo núemro 
    # de niveles 
    pirad_gauss, piramide = PiramideGaussiana (imagen, sigma, incremento, borde, niveles)
    
    # Se calculan los niveles de la pirámide
    for i in range (0, niveles - 1):
        
        # Se extrapola la imagen siguiente a la que se está estudiando
        aux = cv2.pyrUp(pirad_gauss[i+1], (pirad_gauss[i].shape[0], pirad_gauss[i].shape[1]))
        
        # Se calcula la diferencia de la imagen extrapolada y la imagen actual 
        # y se guarda en el conjunto de imágenes de la pirámide
        imagenes.append(DiferenciaGaussianas(pirad_gauss[i], aux))
    
    # Se guarda la última imagen de la pirámide Gaussiana, correspondiente a las
    # frecuencias bajas de la imagen
    imagenes.append(pirad_gauss[-1])      
    
    # Se consigue la imagen compuesta de la pirámide
    piramide = Piramide(imagenes)
    
    # Se devuelve el conjunto de imágenes de la pirámide y la imagen compuesta
    return imagenes, piramide  

# Función para calcular la diferencia de dos imágenes
def DiferenciaGaussianas(img1, img2):
    
    # La imagen resultado tendrá el tamaño de la imagen más pequeña. Esto se 
    # hace porque al extrapolar una imagen puede conseguirse una fila o columna
    # más que la original. 
    if img1.shape[0] < img2.shape[0]:
        x = img1.shape[0]
    
    else:
        x = img2.shape[0]
    
    if img1.shape[1] < img2.shape[1]:
        y = img1.shape[1]
        
    else:
        y = img2.shape[1]
    
    # Se crea la imagen nueva con las dimensiones de la primera imagen
    if img1.ndim == 3:
        img = np.empty((x,y,3))
    
    else:
        img = np.empty((x,y))
    
    # Se restan los elementos de las imágenes
    for i in range (0, x):
        for j in range (0, y):
            img[i,j] = img1[i,j] - img2[i,j]
    
    # Se devuelve la imagen resultado
    return img

# Ejemplo de funcionamiento de la función de pirámide Laplaciana 
def Apartado2B(gris, color, sigma, incremento, niveles, bordes):
    
    print ("    Pirámide Laplaciana: ")
    
    pgrises = []
    pcolores = []
    
    # Se calculan las pirámides laplacianas con las combinaciones de los parámetros
    # y se guardan los resultados
    for b in bordes:
        grises, piragris = PiramideLaplaciana(gris, sigma, incremento, b, niveles)
        colores, piracol = PiramideLaplaciana(color, sigma, incremento, b, niveles)
    
        pgrises.append(piragris)
        pcolores.append(piracol)
    
    print("     -> Parámetros usados : ")
    print("         * Sigma inicial: 0.1\n         * Incremento de sigma:1.2")
    print("         * Bordes: {Reflejado, Negro, Valor replicado}")
    
    titulo = ["Borde Reflejado", "Borde Negro", "Borde Valor Replicado"]
    
    # Se pintan las pirámides obtenidas
    for i in range (0, len(pgrises)):
        
        PintaVarias([pgrises[i]], 1, 1, [str(titulo[i] + " - gris")])
    
    input ("\n\t -> Pulsar ENTER para continuar con las imágenes a color: ")
    
    for i in range (0, len(pcolores)):
        PintaVarias([pcolores[i]], 1, 1, [str(titulo[i] + " - color")])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
# EJERCICIO 2 - APARTADO C

# Función para buscar las regiones relevantes de la imagen
def BusquedaRegiones(imagen, sigma, escalas, incremento, titulo):
    
    imagenes = []
    sigmas = []
    
    # Se crea la matriz que recogerá las áreas seleccionadas
    resultado = np.zeros_like(imagen)
    
    # Se buscan áreas en diferentes niveles
    for k in range (0, escalas):
        
        # Se calcula el tamaño de la máscara en función al valor de sigma
        tam = int(sigma*6 +1)
        if tam%2 == 0:
            tam += 1
        
        # Se hace la convolución de la imagen con una máscara Laplaciana-de-Gaussiana
        img = Laplacian(imagen, tam)
        
        # Se recorre la matriz por áreas buscando los valores máximos de 
        for i in range (0, imagen.shape[0] - tam, tam):
            for j in range (0, imagen.shape[1] - tam, tam):
                
                # Se inicializan los valores para encontrar los máximos locales
                entra = False
                maxi = 0
                
                # Se buscan los máximos locales por zonas de la imagen
                for l in range (0, tam):
                    for m in range (0, tam):
                        
                        if img[i + l, j + m] > maxi:
                            maxi = img[i + l, j + m]
                            cord = (i + l, j + m)
                            
                            entra = True
                
                # Si se ha conseguido un máximo se añade a la matriz resultado
                if entra == True:
                    resultado[cord[0], cord[1]] = maxi
        
        imagenes.append(resultado)
        resultado = np.zeros_like(img)
        
        # Se guarda el sigma utilizado
        sigmas.append(sigma)
        
        # Se incrementa el sigma para el siguiente nivel
        sigma *= incremento
 
    # Se guardan solo los máximos reales
    imagenes = ReduccionNoMaximos(imagenes)
    
    # Se pinta la imagen y las áreas selecionadas
    PintaAreas (imagen, imagenes, sigmas, titulo)
    
    return imagenes

# Función para seleccionar las regiones máximas en cada zona en todas las capas
def ReduccionNoMaximos(imagenes):
    
    # Se recorren todas las filas y columas de las imagenes buscando en que 
    # zona se consigue el valor máximo
    for i in range (0, imagenes[0].shape[0]):
        for j in range (0, imagenes[0].shape[1]):
            
            maxi = imagenes[0][i,j]
            ind = 0
            
            # Se busca cual es la imagen con el máximo valor en ese pixel
            for k in range (1, len(imagenes)):
                
                if imagenes[k][i,j] > maxi:
                    maxi = imagenes[k][i,j]
                    ind = k
            
            # En las imágenes en las que no se tiene el valor máximo se ponen 
            # a 0
            for k in range (0, len(imagenes)):
                
                if k != ind and imagenes[k][i,j] > 0:
                    imagenes[k][i,j] = 0
    
    # Se devuleven las imágenes modificadas
    return imagenes

# Función para pintar la imagen con las áreas que se selccionan al buscar las 
# regiones
def PintaAreas (imagen, imagenes, sigmas, titulo):
    
    # Se ajusta la imagen original
    img = AjustaTribanda(imagen)
    img = NormalizarIntervalo(img)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    c = ['yellow', 'r', 'b', 'g', 'pink', 'violet']
    # Título que se le va a poner a la imagen
    plt.title(titulo)
    
    # Se muestra la imagen original
    plt.imshow(img)
    
    # Se pintan las áreas en función al sigma de cada uno
    for k in range (0, len(imagenes)):
        
        # Se calcula el tamaño asociado al sigma
        tam = (int)(sigmas[k]*6 + 1)
        
        if tam%2 == 0:
            tam += 1
        
        x = []
        y = []
        
        # Se buscan las regiones encontradas con este sigma
        for i in range (0, imagenes[k].shape[0]):
            for j in range (0, imagenes[k].shape[1]):
                
                if imagenes[k][i,j] > 0:
                    x.append(j)
                    y.append(i)
        
        # se pintan las regiones
        plt.scatter(x, y, s=tam, facecolors='none', edgecolors=c[k%len(c)], label="sigma: " + str(sigmas[k]))
    
    # Se pinta la imagen
#    plt.legend()
    plt.show()

# Función para comprar dos conuntos de regiones distintos. 
def ComparacionRegiones (imagen, imagenes1, imagenes2):

    # Se crean las matrices que se van a utilizar para comparar
    img1 = np.zeros_like(imagenes1[0])
    img2 = np.zeros_like(imagenes2[0])
    
    # Se recorre cada capa y se guardan todas las áreas encontradas en una misma 
    # matriz
    for k in range (0, len(imagenes1)):
        for i in range (0, imagenes1[0].shape[0]):
            for j in range (0, imagenes1[0].shape[0]):
                
                if imagenes1[k][i,j] > 0:
                    img1[i,j] = imagenes1[k][i,j]
                
                if imagenes2[k][i,j] > 0:
                    img2[i,j] = imagenes2[k][i,j]
    
    # Se comparan las dos matrices y se guardan las áreas que solo aparezcan en
    # una de ellas
    for i in range (0, img1.shape[0]):
        for j in range (0, img1.shape[1]):
            
            if img1[i,j] > 0 and img2[i,j] > 0:
                img1[i,j] = 0
                img2[i,j] = 0
    
    # Se pintan los resultados conseguidos
    PintaAreas (imagen, [img1, img2], [1.2, 1.4], "Comparacion de regiones")
    
# Función para probar el algoritmo de búsqueda de regiones con distintos 
# incrementos de sigma
def Apartado2C(gris, sigma, incremento, niveles):
    
    print ("    Búsqueda de regiones: ")
    
    print("     -> Parámetros usados : ")
    print("         * Sigma inicial: 1\n         * Incremento de sigma: {1.2, 1.4}")
    print("         * Escalas: 4")
    
    regiones = []
    # Se calculan las regiones y se pintan
    for i in incremento:
        regiones.append(BusquedaRegiones(gris, sigma, niveles, i, str(str(niveles) + " niveles e incremento de " + str(i))))

    ComparacionRegiones (gris, regiones[0], regiones[1])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
# EJERCICIO 3

# Función para hibridar las dos imágenes pasadas como argumento    
def Hibridas (altas, bajas, tam_alta, tam_baja):
    
    # La imagen hibrida tendrá el tamao de la imagen más pequeña
    if bajas.shape[0] < altas.shape[0]:
        x = bajas.shape[0]
        
    else:
        x = altas.shape[0]
    
    if bajas.shape[1] < altas.shape[1]:
        y = bajas.shape[1]
    else:
        y = altas.shape[1]
    
    # Se crea la imagen hibrida vacía con las dimensiones de las originales
    if bajas.ndim == 3:
        img = np.empty((y,x,3))
    
    else:
        img = np.empty((y,x))
    
    # se escalan las imagenes al mismo tamaño
    img_baja = cv2.resize(bajas, (x,y), interpolation = cv2.INTER_AREA)
    img_alta = cv2.resize(altas, (x,y), interpolation = cv2.INTER_AREA)
    
    # Se consigue la máscara para aplicar la convolución a la imagen que se 
    # quedará con frecuencias bajas
    kb = cv2.getGaussianKernel(tam_baja, (tam_baja - 1)/6)
    
    # Se aplica la convolución 1D a la imagen 1 y se normalizan los valores 
    # conseguidos
    img_baja = Convolution1D (img_baja, kb, kb.transpose())
    img_baja = NormalizarIntervalo(img_baja)
    
    # Se aplica la Laplaciana a la imagen 2 y se normaluzan los valores conseguidos
    img_alta = Laplacian(img_alta, tam_alta)
    img_alta = NormalizarIntervalo(img_alta)
    
    # Se suman los valores de los resultados de las dos imágenes para conseguir
    # la imagen hibridada
    for i in range (0, y):
        for j in range (0, x):
            img[i,j] = img_baja[i,j] + img_alta[i,j]
    
    # Se devuelve la imagen hibridada
    return img 

# Ejemplo de funcionamiento de las imágenes hibridadas 
def Apartado3(altas, bajas, tam_altas, tam_bajas, titulos, sigma, niveles, incremento):
    
    print ("    Imágenes hibridas: ")
    
    for i in range (0, len(altas)):
        hibrida = Hibridas(altas[i], bajas[i], tam_altas[i], tam_bajas[i])
        
        aux, piramide = PiramideGaussiana(hibrida, sigma, incremento, cv2.BORDER_REFLECT_101, niveles)
        PintaMI([altas[i], hibrida, bajas[i]], titulos[i], 'PLT')
        PintaVarias([piramide], 1, 1, ["Piramide Gaussiana: " +  titulos[i]])

# Ejercicio del Bonus: 2
def ApartadoBonus2 (altas, bajas, tam_altas, tam_bajas, titulos, sigma, niveles, incremento):
     
    print ("    Ejercicio Bonus 2 - Imágenes hibridas: ")
    
    
    for i in range (0, len(altas), 2):
                    
        hibrida1 = Hibridas(altas[i], bajas[i], tam_altas, tam_bajas)
        hibrida2 = Hibridas(altas[i + 1], bajas[i + 1], tam_altas, tam_bajas)
        
        aux, piramide1 = PiramideGaussiana(hibrida1, sigma, incremento, cv2.BORDER_REFLECT_101, niveles)
        aux, piramide2 = PiramideGaussiana(hibrida2, sigma, incremento, cv2.BORDER_REFLECT_101, niveles)
        PintaVarias([piramide1], 1, 1, ["(Grises)" + titulos[i//2], titulos[i//2]])
        PintaVarias([piramide2], 1, 1, ["(color)" + titulos[i//2], titulos[i//2]])
        
        if i != (len(altas) - 2):
            input ("\n\t -> Pulsar ENTER para continuar con el siguiente conjunto de imagenes: ")

##############################################################################
###########################  FUNCIONES AUXILIARES  ###########################
##############################################################################
       
def LeeImagenCV (filename, flagColor):
    
    img = cv2.imread(filename, flagColor)
    
    img = img.astype(np.float)
    
    return img

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
    
# Función para visualizar varias imágenes en una sola ventana y cada una con 
# su título
def PintaVarias (vim, x, y, titles):       
        
    # Contador para el id de la imagen
    cont = 1
    
    # Se recorren todas las imagenes de la lista y se pintan
    for i in range (len(vim)):
        
        vim[i] = AjustaTribanda(vim[i])
        vim[i] = NormalizarIntervalo(vim[i])
        vim[i] = vim[i].astype(np.uint8)
        vim[i] = cv2.cvtColor(vim[i], cv2.COLOR_BGR2RGB)
        
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

def PintaMI (vim, titulo='Conjunto', pinta='CV'):
    
    # Si se pinta con cv2, las imágenes se escalan a la imagen más pequeña, ya
    # que si se escala a la mayor cv2 muestra lo que saca por pantalla. 
    if (pinta == 'CV'):
        
        # Se busca la imagen más pequeña para escalar el resto a ella
        max_size = 999999999999
            
        for i in range (0, len(vim)):
            
            # Se aprovecha para comprobar que todas las imágenes sean tribanda
            vim[i] = AjustaTribanda(vim[i])
            vim[i] = NormalizarIntervalo (vim[i])
            
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
            vim[i] = NormalizarIntervalo (vim[i])
            
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
        cv2.imshow(titulo, imagen)
    
    else:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        plt.title(titulo)
        plt.imshow(imagen)
        plt.show()

def PintarImagen(imagen, titulo):
    
    img = AjustaTribanda(imagen)
    
    img = NormalizarIntervalo(img)
    
    img = img.astype(np.uint8)
    
    cv2.imshow(titulo, img)

##############################################################################
############################  PROGRAMA PRINCIPAL  ############################
##############################################################################

def main():

    print ("\nEjercicio 1:")
    print ("\n -> Apartado A:")

    # Parámetros para el apartado A
    gris = LeeImagenCV("imagenes/bicycle.bmp", 0)
    color = LeeImagenCV("imagenes/motorcycle.bmp",1)
    
    tam = [3, 31]
    sig = [0.5, 5.17]
    bor = [cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT]
    
    Apartado1A(gris, color, tam, sig, bor)

    input ("\nEnter para continuar con el Apartado B:")
    
    print ("\n -> Apartado B:")
    
    # Parámetros para el apartado B
    gris = LeeImagenCV("imagenes/cat.bmp", 0)
    color = LeeImagenCV("imagenes/dog.bmp",1)
    
    tam = [7, 13, 19]
    bor = [cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT]
    
    Apartado1B(gris, color, tam, bor)
    
    input ("\nEnter para continuar con el Ejercicio 2:")
    
    print ("\nEjercicio 2:")
    print ("\n -> Apartado A:")
    
    # Parámetros usados para el apartado A
    gris = LeeImagenCV("imagenes/plane.bmp", 0)
    color = LeeImagenCV("imagenes/bird.bmp",1)
    
    sigma = 0.1
    incremento = 1.2
    niveles = 4
    bordes = [cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
    
    Apartado2A(gris, color, sigma, incremento, niveles, bordes)
    
    input ("\nEnter para continuar con el Apartado B:")
    
    print ("\n -> Apartado B:")
    
    # Parámetros del apartado B
    gris = LeeImagenCV("imagenes/plane.bmp", 0)
    color = LeeImagenCV("imagenes/bird.bmp",1)
    
    sigma = 0.1
    incremento = 1.2
    niveles = 4
    bordes = [cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
    
    Apartado2B(gris, color, sigma, incremento, niveles, bordes)
    
    input ("\nEnter para continuar con el Apartado C:")
    
    print ("\n -> Apartado C:")
    
    # Parámetros del apartado C
    gris = LeeImagenCV("imagenes/fish.bmp", 0)
    
    sigma = 1
    incremento = [1.2, 1.4]
    
    niveles = 5
    
    Apartado2C(gris, sigma, incremento, niveles)
    
    input ("\nEnter para continuar con el Ejercicio 3:")
    
    print ("\nApartado 3:")
    
    # Parámetros del ejercicio 3
    altas = [LeeImagenCV("imagenes/plane.bmp", 0), LeeImagenCV("imagenes/cat.bmp",0), 
             LeeImagenCV("imagenes/submarine.bmp",0)]
    
    bajas = [LeeImagenCV("imagenes/bird.bmp", 0), LeeImagenCV("imagenes/dog.bmp",0),
             LeeImagenCV("imagenes/fish.bmp",0)]
    
    tam_altas = [31, 31, 31]
    tam_bajas = [21, 21, 21]
    
    titulos = ["Avion - Pajaro", "Gato - Perro", "Submarino - Pez"]
    sigma = 0.5
    niveles = 4
    incremento = 0.5
    
    Apartado3(altas, bajas, tam_altas, tam_bajas, titulos, sigma, niveles, incremento)
    
    input ("\nEnter para continuar con el Bonus - Ejercicio 2:")
    
    print ("\nBunus Apartado 2:")
    
    # Parámetros del bonus ejercicio 2
    altas = [LeeImagenCV("imagenes/plane.bmp", 0), LeeImagenCV("imagenes/plane.bmp", 1),
             LeeImagenCV("imagenes/cat.bmp",0), LeeImagenCV("imagenes/cat.bmp",1), 
             LeeImagenCV("imagenes/submarine.bmp",0), LeeImagenCV("imagenes/submarine.bmp",1), 
             LeeImagenCV("imagenes/bicycle.bmp",0), LeeImagenCV("imagenes/bicycle.bmp",1), 
             LeeImagenCV("imagenes/einstein.bmp",0), LeeImagenCV("imagenes/einstein.bmp",1)]
    
    bajas = [LeeImagenCV("imagenes/bird.bmp", 0), LeeImagenCV("imagenes/bird.bmp", 1),
             LeeImagenCV("imagenes/dog.bmp",0), LeeImagenCV("imagenes/dog.bmp",1),
             LeeImagenCV("imagenes/fish.bmp",0), LeeImagenCV("imagenes/fish.bmp",1),
             LeeImagenCV("imagenes/motorcycle.bmp",0), LeeImagenCV("imagenes/motorcycle.bmp",1),
             LeeImagenCV("imagenes/marilyn.bmp",0), LeeImagenCV("imagenes/marilyn.bmp",1)]
    
    tam_altas = 31
    tam_bajas = 21
    
    titulos = ["Avion - Pajaro", "Gato - Perro", "Submarino - Pez", "Bicicleta - Moto", "Einstein - Marilyn"]
    sigma = 0.5
    niveles = 4
    incremento = 0.5
    
    ApartadoBonus2(altas, bajas, tam_altas, tam_bajas, titulos, sigma, niveles, incremento)
    
    

##############################################################################
##############################################################################
    
# Se llama al programa principal para que ejecute el programa
if __name__ == '__main__':
    main()
    