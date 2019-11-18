# -*- coding: utf-8 -*-
#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

import numpy as np
import keras
import matplotlib.pyplot as plt 
import keras.utils as np_utils

# Modelos y capas que se van a usar
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Activation

# Optimizador que se va a usar
from keras.optimizers import SGD 

# Preprocesamiento de datos
from keras.preprocessing.image import ImageDataGenerator

# Conjunto de datos
from keras.datasets import cifar100

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# Esta función solo se le llama una vez. Devuelve 4 vectores conteniendo:
#    -> Las imágenes de entrenamiento
#    -> Las clases de las imágenes de entrenamiento
#    -> Las imágenes del conjunto de test
#    -> Las clases del conunto de test
# (En ese orden)

def cargarImagenes():
    
    # Cargamos Cifrar100 
    # Cada imagen tienen tamaño (32,32,3)
    # Nos quedamos con 25 clases
    
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Se normaliza entre 0 y 1
    x_train /= 255
    x_test /= 255
    
    # Nos quedamos con 25 clases
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape(train_idx, -1)
    
    # Guardamos los conjuntos de esas 25 clases
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    
    # Nos quedamos con 25 clases
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    
    # Guardamos los conjuntos de esas 25 clases
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    # Transformamos los vectores de clases en matrices. 
    # Cada componente se convierte en un vecor de ceros con un uno en el 
    # componente correspondiente a la clase a la que pertenece la imagen. Este 
    # paso es NECESARIO para la clasificación multiclase en keras. 
    
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)
    
    return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el porcentaje 
# de etiquetas bien predichas frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas y el vector de etiquetas 
# predichas, en el formato de keras (matrices donde cada etiqueta ocupa una 
# fila, con un 1 en la posición de la clase a la que pertenece y 0 en las demás)

def calcularAccuracy(labels, preds):
    
    labels = np.argmax(labels, axis=1)
    preds = np.argmax(preds, axis=1)
    
    accuracy = sum(labels == preds)/len(labels)
    
    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función de 
# pérdida en el conjunto de train y en el de validación, y otra con la 
# evolución del accuracy en el conjunto de train y el de validación. Es 
# necesario pasarle como parámetro el historial del entrenamiento del modelo
# (lo que devuelven las funciones fit() y fit_generator())

def mostrarEvolucion(hist):
    
    loss = hist.history['loss']
    
    val_loss = hist.history['val_loss']
    
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    
    acc = hist.history['acc']
    
    val_acc = hist.history['val_acc']
    
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

x_train, y_train, x_test, y_test = cargarImagenes()

model = Sequential()

model.add(Conv2D(6, (5,5), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())    # Aplana la matriz

model.add(Dense(50, activation='linear'))     # 50 es la dimensión de salida
model.add(Activation('relu'))

model.add(Dense(25, activation='linear'))
model.add(Activation('softmax'))              # Transformar la salida de las neuronas 
                                              # en la probabilidad de pertenecer a cada clase
model.summary()

print()

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=SGD(), metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

#print("\n Modelo basico \n\n")
#historial = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1,
#                      validation_data=(x_train, y_train))

# 20 - 25 épocas
# 32 de batch

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

#score = model.evaluate(x_test, y_test, verbose=0)

#print('\nTest loss: %f' % (score[0]))
#print('Test accuracy: %f' % (score[1]))

#mostrarEvolucion(historial)

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

print("\n\n Mejora del modelo")

# Aumento de datos (+ Normalización)
print ("\n -> Normalizacion + Aumento de datos\n")

datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,
                             zoom_range=0.2, horizontal_flip=True, rotation_range=20)
datagen.fit(x_train)

historial = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),  
                                verbose=1, epochs=20, 
                      validation_data=datagen.flow(x_train, y_train))

score = model.evaluate(x_test, y_test, verbose=0)

print('\nTest loss: %f' % (score[0]))
print('Test accuracy: %f' % (score[1]))

mostrarEvolucion(historial)