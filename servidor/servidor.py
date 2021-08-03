from flask import Flask
from flask import request
import base64
import cv2
import numpy as np
from Prediccion import Prediccion
from time import process_time

dirImg = "CropsServidor"
extImg = ".png"
clases = ["Martillo", "Destornillador", "Llave", "Alicate", "Regla"]
modelos_todos = []
for i in range(1, 4):
    modelos_todos.append(Prediccion("models/modelo"+str(i)+".h5", 256, 256))

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
            tiempo = 0
            total = 0
            for img in imagenes:
                decode_image(img["content"], img["id"])
                imagen = cv2.imread(dirImg + "/" + img["id"])
                modelo0 = modelos_todos[0]
                num = int(modelo) - 1
                if num < len(modelos_todos) and num >= 0:
                    modelo0 = modelos_todos[num]
                t1_start = process_time()
                claseResultado = modelo0.predecir(imagen)
                t1_stop = process_time()
                print(modelo)
                print(clases[claseResultado])
                resultados.append({"class": clases[claseResultado], "id-image": img["id"]})
                tiempo = tiempo + t1_stop - t1_start
                total = total + 1
            if total != 0:
                tiempo = tiempo / total
            resultadosTodos.append({"model_id":modelo, "results":resultados})

        return {"status":"success", "message":"Predictions made satisfactorily", "results":resultadosTodos}, 200
    return {"status":"error", "message":"Error making predictions"}, 400


def decode_image(im_b64, fileName):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    cv2.imwrite(dirImg + '/' + fileName, img)

if __name__=="__main__":
    app.run(debug=True,port=8000)