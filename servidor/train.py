import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten



#Carga imagenes según categorias por carpetas
def cargarDatos(fase,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)

            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados


print("tensorflow", tf.__version__)
print("keras", keras.__version__)

ancho = 256
alto = 256
pixeles=ancho*alto
#Imagen RGB --> 3 Canales
#Blanco y negro --> 1 Canal
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=10

cantidadDatosEntrenamiento=[60, 60, 60, 60, 60]
cantidadDatosPruebas=[20, 20, 20, 20, 20]

#Cargar las imágenes
imagenes, probabilidades = cargarDatos("dataset/train/",numeroCategorias,cantidadDatosEntrenamiento,ancho,alto)

model=Sequential()

#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

#Capas ocultas
#Capas convolucionales
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
#Convolución reduce, maxpool también reduce. Para otras capas puedo copiar y pegar
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))

#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
model.fit(x=imagenes, y=probabilidades, epochs=35, batch_size=60)

imagenesPrueba, probabilidadesPrueba = cargarDatos("dataset/test/",numeroCategorias,cantidadDatosPruebas,ancho,alto)
resultados = model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=", resultados[1])
#0 loss(perdida), 1 accuracy

# Guardar modelo
ruta = "models/modelo0.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()