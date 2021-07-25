from flask import Flask
from flask import request
import base64
import cv2
import numpy as np
#from Prediccion import Prediccion

dirImg = "CropsServidor"
extImg = ".png"
clases = ["Martillo", "Destornillador", "Llave", "Alicate", "Regla"]

app = Flask(__name__)

#http://127.0.0.1:8000/predict
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        cliente = request.json["id_client"]
        imagenes = request.json["images"]
        modelos = request.json["models"]

        resultadosTodos = []
        if len(modelos) == 0 or len(modelos) == 1 and modelos[0] == "":
            modelos = [0]
        for modelo in modelos:
            resultados = []
            for img in imagenes:
                decode_image(img["content"], img["id"])
                #miModeloCNN = Prediccion("models/modelo'+modelo+'.h5", 256, 256)
                #imagen = cv2.imread(dirImg + "/" + img["id"] + extImg)
                #claseResultado = miModeloCNN.predecir(imagen)
                claseResultado = "Martillo"
                resultados.append({"class":claseResultado, "id-image":img["id"]})
            resultadosTodos.append({"model_id":modelo, "results":resultados})

        return {"status":"success", "message":"Predictions made satisfactorily", "results":resultadosTodos}, 200
    return {"status":"error", "message":"Error making predictions"}, 400


def decode_image(im_b64, fileName):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    cv2.imwrite(dirImg + '/' + fileName + extImg, img)

if __name__=="__main__":
    app.run(debug=True,port=8000)