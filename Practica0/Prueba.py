# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:56:18 2019

@author: Laura
"""

import cv2
from matplotlib import pyplot as plt 

dave = cv2.imread('dave.jpg',0)
logo = cv2.imread('logoOpenCV.jpg',0)
messi = cv2.imread('messi.jpg',0)
orapple = cv2.imread('orapple.jpg',1)

###FLAG -> 0 (blanco y negro) 1 (color)
#cv2.imshow('Dave',dave)
#cv2.imshow('Logo OpenCV',logo)
#cv2.imshow('Messi',messi)
#cv2.imshow('Orapple',orapple)
#
## Líneas necesarias para mostrar las imágenes
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imgrgb = cv2.cvtColor(messi, cv2.COLOR_BGR2RGB) # Cambiar de BGR a RGB
logogray = cv2.cvtColor(logo, cv2.COLOR_GRAY2RGB) 
##

plt.imshow(imgrgb)
plt.title('Prueba')
plt.show()
#
#plt.imshow(messi)
#plt.show()

plt.imshow(logogray)
plt.show() 

plt.imshow(logo)
plt.show()
# OpenCV no las mantiene en color, BGR no RGB 


# Ejercicio 2: encajar entre 0 y 255 y mostrar
# Ejercicio 3: juntar varias imagenes en una sola y adapatar los canales de cada 
#              una para el mimso (3 canales)