
import PIL
import io
import base64
import re
from io import StringIO
from PIL import Image
import os
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from datetime import datetime

classes = ["ウソップ", "サンジ", "ジンベエ", "ゾロ", "チョッパー",
           "ナミ", "ビビ", "フランキー", "ブルック", "ルフィ", "ロビン"]
classes_img = ["static/img/usoppu.jpg", "static/img/sanji.jpg", "static/img/jinbe.jpg", "static/img/zoro.jpg", "static/img/chopper.jpg",
               "static/img/nami.jpg", "static/img/bibi.jpg", "static/img/franky.jpg", "static/img/bruck.jpg", "static/img/rufi.jpg", "static/img/robin.jpg"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = "./static/image/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model('./one-piece_cnn_aug.h5')  # 学習済みモデルをロードする

graph = tf.get_default_graph()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    global graph
    with graph.as_default():
        if request.method == 'POST':

            myfile = request.form['snapShot'].split(',')
            imgdata = base64.b64decode(myfile[1])
            image = Image.open(io.BytesIO(imgdata))

            # 保存
            basename = datetime.now().strftime("%Y%m%d-%H%M%S")
            image.save(os.path.join(UPLOAD_FOLDER, basename+".png"))
            filepath = os.path.join(UPLOAD_FOLDER, basename+".png")

            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X).astype('float32')
            X /= 256

            result = model.predict([X])[0]
            result_proba = model.predict_proba(X, verbose=1)[0]

            percentage = (result_proba*100).astype(float)
            array_sort = sorted(
                list(zip(percentage, classes, classes_img)), reverse=True)
            percentage, array_class, array_img = zip(*array_sort)

            pred_answer1 = "ラベル:" + \
                str(array_class[0]) + ",確率："+str(percentage[0])+"%"
            pred_answer2 = "ラベル:" + \
                str(array_class[1]) + ",確率："+str(percentage[1])+"%"
            pred_answer3 = "ラベル:" + \
                str(array_class[2]) + ",確率："+str(percentage[2])+"%"

            img_src1 = array_img[0]
            img_src2 = array_img[1]
            img_src3 = array_img[2]

            basename = datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath3 = UPLOAD_FOLDER + basename+".png"

            return render_template("index.html", answer1=pred_answer1, img_data1=img_src1, answer2=pred_answer2, img_data2=img_src2, answer3=pred_answer3, img_data3=img_src3)

        return render_template("index.html", answer="")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
    #app.run(debug = True)
