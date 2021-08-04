import cv2
import os
import base64
import requests
import numpy as np

nameWindow="Proyecto_"
def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,45,255,nothing)
    cv2.createTrackbar("max", nameWindow, 35, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 6,30, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 15000, 10000, nothing)
    #cv2.createTrackbar("areaMax", nameWindow, 5000, 100000000, nothing)

def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))

    sorted_index = [i[0] for i in sorted(enumerate(areas), key=lambda x:x[1])]
    sorted_index.reverse()
    return areas, sorted_index

def detectarForma(imagen):
    imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    img = imagen.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    cv2.imshow('a_channel', a)
    imagenGris = a
    #cv2.imshow("Grises",imagenGris)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    bordes = cv2.Canny(imagenGris, min, max)
    kernel = np.ones((tamañoKernel,tamañoKernel),np.uint8)
    bordes = cv2.dilate(bordes,kernel)
    cv2.imshow("Bordes",bordes)
    figuras,jerarquia = cv2.findContours(bordes, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    areas, sorted_index = calcularAreas(figuras)
    areaMin=cv2.getTrackbarPos("areaMin", nameWindow)
    primer_figura = None
    primer_vertices = None
    mensaje = "ROI"
    for indice_figura in sorted_index:
        if areas[indice_figura] >= areaMin:
            figuraActual = figuras[indice_figura]
            vertices = cv2.approxPolyDP(figuraActual,0.05*cv2.arcLength(figuraActual,True),True)
             #str(len(vertices))
            if len(vertices) == 4 and jerarquia[0][indice_figura][3]!=-1:
                if primer_figura is None:
                    primer_figura = figuraActual
                    primer_vertices = vertices
                else:
                    cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
                    return vertices
    if primer_figura is not None:
        cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawContours(imagen, [primer_figura], 0, (0, 0, 255), 2)
        return primer_vertices




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
    req = {"id":file, "content":im_b64.decode('ascii')}
    return req


def format_request(imgs):
    datos = {"id_client": "001",
             "images": imgs,
             "models": [1, 2, 3, 4]}
    print(datos)
    return datos


def send_request(request):
    try:
        response = requests.post('http://localhost:8000/predict', json=request)
        print("Status code: ", response.status_code)
        print("Printing Entire Post Request")
    except:
        print("Couldn't establish connection")
        return
    if response.status_code == 200 or response.status_code == 400:
        print(response.json())
        show_results(response.json())
        import os
        directories = os.listdir('Crops/')
        for file in directories:
            os.remove('Crops/' + str(file))
    else:
        print("Unexpected server's error")

def show_results(json):
    data = json['results']
    parsed_data = {}
    for resultado in data:
        for info in resultado['results']:
            if parsed_data.get(info['id-image']) is None:
                parsed_data[info['id-image']] = []
            info['model_id'] = resultado['model_id']
            parsed_data[info['id-image']].append(info)
    print(parsed_data)
    for imagen in parsed_data:
        info = parsed_data[imagen]
        img = cv2.imread('Crops/' + imagen)
        for modelo in info:
            mensaje = "Modelo "+str(modelo['model_id'])+" Clase: "+ modelo['class']
            cv2.putText(img, mensaje, (10, 50+35*modelo['model_id']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(str(modelo['id-image']), img)
    #{'message': 'Predictions made satisfactorily', 'results': [{'model_id': 0, 'results': [{'class': 'Regla', 'id-image': 'crop_0.png'}]}], 'status': 'success'}


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
        try:
            saved = save_image(imagen_pre,coordenadas,numero)
            #cv2.imshow("Imagen Guardada " + str(numero), saved)
            numero+=1
        except:
            print("No hay ROI detectada")
        #save


cv2.destroyAllWindows()