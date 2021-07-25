from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

class Prediccion():
    def __init__(self,ruta,ancho,alto):
        self.modelo = load_model(ruta)
        self.alto = alto
        self.ancho = ancho

    def predecir(self, imagen):
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas = []
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA = np.array(imagenesCargadas)
        predicciones=self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones",predicciones)
        clasesMayores=np.argmax(predicciones,axis=1)
        return clasesMayores[0]