import cv2
import os
import base64
import requests
import numpy as np

nameWindow="Proyecto"
def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,200,255,nothing)
    cv2.createTrackbar("max", nameWindow, 250, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 8,30, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)
    #cv2.createTrackbar("areaMax", nameWindow, 5000, 100000000, nothing)

def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

def detectarForma(imagen):
    imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grises",imagenGris)
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
            mensaje = "ROI" #str(len(vertices))
            if len(vertices) == 4:
                cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
                return vertices


        i=i+1


def save_image(image, contours, num):
    print(contours)
    x,y,w,h = cv2.boundingRect(contours)
    new_img=image[y:y+h,x:x+w]
    cv2.imwrite('Crops/'+'crop_'+str(num)+ '.png', new_img)
    return new_img


def send_server():
    directories = os.listdir('Crops/')
    convertidas = []
    for file in directories:
        convertidas.append(convert_image(file))

    request = format_request(convertidas)
    send_request(request)
    pass


def convert_image(file):
    img = cv2.imread('Crops/'+file)
    _, im_arr = cv2.imencode('.png', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    req = {"id":file, "content":im_b64}
    return req


def format_request(imgs):
    datos = {"id_client": "001",
             "images": imgs,
             "models": [""]}
    print(datos)
    return datos


def send_request(request):
    response = requests.post('https://localhost:8080/predict', json=request)
    print("Status code: ", response.status_code)
    print("Printing Entire Post Request")
    print(response.json())


### MAIN ###
video = cv2.VideoCapture(0)
constructorVentana()
numero=0
while True:
    _, imagen = video.read()
    imagen_pre = imagen.copy()
    coordenadas = detectarForma(imagen)
    cv2.imshow("Imagen", imagen)
    #Para el programa
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
    elif k==101:
        print("e")
        send_server()
    elif k==99:
        print("c")
        saved = save_image(imagen_pre,coordenadas,numero)
        cv2.imshow("Imagen Guardada " + str(numero), saved)
        numero+=1
        #save


cv2.destroyAllWindows()