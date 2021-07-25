import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from sklearn.model_selection import KFold
from keras import backend as K
from sklearn.metrics import classification_report


numero_modelo = 1

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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
numeroCategorias=5

cantidadDatosEntrenamiento=[80, 80, 80, 80, 80]
cantidadDatosPruebas=[20, 20, 20, 20, 20]

#Cargar las imágenes
imagenes, probabilidades = cargarDatos("dataset/train/",numeroCategorias,cantidadDatosEntrenamiento,ancho,alto)



imagenesPrueba, probabilidadesPrueba = cargarDatos("dataset/test/",numeroCategorias,cantidadDatosPruebas,ancho,alto)
#resultados = model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
#print("Accuracy=", resultados[1])
#0 loss(perdida), 1 accuracy


inputs = np.concatenate((imagenes, imagenesPrueba), axis=0)
targets = np.concatenate((probabilidades, probabilidadesPrueba), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)
per_fold = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    model = Sequential()

    # Capa entrada
    model.add(InputLayer(input_shape=(pixeles,)))
    model.add(Reshape(formaImagen))

    # Capas ocultas
    # Capas convolucionales
    model.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
    # Convolución reduce, maxpool también reduce. Para otras capas puedo copiar y pegar
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Aplanamiento
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))

    # Capa de salida
    model.add(Dense(numeroCategorias, activation="softmax"))

    # Traducir de keras a tensorflow

    if fold_no == 1:
        model2 = model
        model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        ruta = "models/modelo" + str(numero_modelo) + ".h5"
        model2.fit(inputs[train], targets[train], epochs=1, batch_size=60)
        model2.save(ruta)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m])


    # Entrenamiento
    res = model.fit(inputs[train], targets[train], epochs=1, batch_size=60)



    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    loss, accuracy, f1_score, precision, recall = model.evaluate(inputs[test], targets[test], verbose=0)
    per_fold.append([loss, accuracy, f1_score, precision, recall])

    # Increase fold number
    fold_no = fold_no + 1

nombre_metricas = ["loss", "accuracy", "f1_score", "precision", "recall"]
total = 0
loss = 0.0
accuracy = 0.0
f1_score = 0.0
precision = 0.0
recall = 0.0
for i in per_fold:
    total = total + 1
    loss = loss + i[0]
    accuracy = accuracy + i[1]
    f1_score = f1_score + i[2]
    precision = precision + i[3]
    recall = recall + i[4]
loss = loss / total
accuracy = accuracy / total
f1_score = f1_score / total
precision = precision / total
recall = recall / total
print("Loss", loss)
print("Accuracy", accuracy)
print("F1 Score", f1_score)
print("Precision", precision)
print("Recall", recall)



# Guardar modelo
# Informe de estructura de la red
#model.summary()