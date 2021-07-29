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
import imutils

numero_modelo = 3

epocas = 8
batch = 60
kernel = [5, 3]
strides = [1, 1]
filtros = [4, 8]
pool = [2, 2]
strides_pool = [2, 2]
padding = ["same", "same"]
activacion = ["relu", "relu"]
capa_densa = 128

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

def formatear_imagen(imagen, ancho, alto):
    imagen2 = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen2 = cv2.resize(imagen2, (ancho, alto))
    imagen2 = imagen2.flatten()
    imagen2 = imagen2 / 255
    return imagen2

#Carga imagenes según categorias por carpetas
def cargarDatos(fase,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagenesCargadas.append(formatear_imagen(imagen, ancho, alto))
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
            for angle in np.arange(0, 360, 10):
                imagen2 = imutils.rotate_bound(imagen, angle)
                imagenesCargadas.append(formatear_imagen(imagen2, ancho, alto))
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria] = 1
                valorEsperado.append(probabilidades)

    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados


print("tensorflow", tf.__version__)
print("keras", keras.__version__)

ancho = 128
alto = 128
pixeles=ancho*alto
#Imagen RGB --> 3 Canales
#Blanco y negro --> 1 Canal
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)

cantidadDatosEntrenamiento=[]
cantidadDatosPruebas=[]
numeroCategorias = 10
for i in range(0, numeroCategorias):
    cantidadDatosEntrenamiento.append(8)
    cantidadDatosPruebas.append(2)

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
    #print(targets[test])
    pos_uno = []
    for i in targets[test]:
        for index, j in enumerate(i):
            if j == 1.:
                pos_uno.append(index)
    #print(pos_uno)

    model = Sequential()

    # Capa entrada
    model.add(InputLayer(input_shape=(pixeles,)))
    model.add(Reshape(formaImagen))

    # Capas ocultas
    # Capas convolucionales
    for k in range(0, len(kernel)):
        model.add(Conv2D(kernel_size=kernel[k], strides=strides[k], filters=filtros[k], padding=padding[k], activation=activacion[k], name="capa_" + str(k)))
        # Convolución reduce, maxpool también reduce. Para otras capas puedo copiar y pegar
        model.add(MaxPool2D(pool_size=pool[k], strides=strides_pool[k]))

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
        res = model2.fit(inputs[train], targets[train], epochs=epocas, batch_size=batch)
        model2.save(ruta)
        prediction = model2.predict(inputs[test])
        pos_uno_p = []
        #print(prediction)
        for i in prediction:
            max = 0
            indexMax = 0
            for index, j in enumerate(i):
                if j > max:
                    max = j
                    indexMax = index
            pos_uno_p.append(indexMax)
        #print(pos_uno_p)
        confusion = tf.math.confusion_matrix(labels=pos_uno, predictions=pos_uno_p, num_classes=numeroCategorias)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc',f1_m,precision_m, recall_m])


    # Entrenamiento
    if fold_no != 1:
        res = model.fit(inputs[train], targets[train], epochs=epocas, batch_size=batch)




    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    loss, accuracy, f1_score, precision, recall = model.evaluate(inputs[test], targets[test], verbose=0)
    per_fold.append([loss, accuracy, f1_score, precision, recall])
    print(per_fold[fold_no - 1])

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
if (total != 0):
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


print(confusion)

# Guardar modelo
# Informe de estructura de la red
#model.summary()