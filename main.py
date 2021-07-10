import cv2
import numpy as np

nameWindow="Calculadora Canny"
def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)
    #cv2.createTrackbar("areaMax", nameWindow, 5000, 100000000, nothing)

def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

def detectarForma(imagen):
    imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grises",imagenGris)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    bordes = cv2.Canny(imagenGris, min, max)
    kernel = np.ones((tamañoKernel,tamañoKernel),np.uint8)
    bordes = cv2.dilate(bordes,kernel)
    cv2.imshow("Bordes",bordes)
    figuras,jerarquia = cv2.findContours(bordes, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    i = 0
    areaMin=cv2.getTrackbarPos("areaMin", nameWindow)
    for figuraActual in figuras:
        if areas[i] >= areaMin:
            vertices = cv2.approxPolyDP(figuraActual,0.05*cv2.arcLength(figuraActual,True),True)
            mensaje = str(len(vertices))
            if len(vertices) == 3:
                #print("Triángulo!")
                mensaje = "Triangulo "+str(areas[i])

            #cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
        i=i+1

video = cv2.VideoCapture('img/video.mp4')
constructorVentana()
while True:
    _,imagen = video.read()
    detectarForma(imagen)
    cv2.imshow("Imagen", imagen)

    #Para el programa
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()