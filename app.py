import os
from flask import Flask, request, render_template
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model


app = Flask(__name__)

classes = ['Eczema', 'Vascular Tumors', 'Bullous Disease', 'Nail Fungus']

new_model = load_model('Static/model/skin_train.h5')

def predict(path):
    new_model.summary()
    test_image = keras.utils.load_img(path,target_size=(224,224))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    result = new_model.predict(test_image)
    return result[0]
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["GET", "POST"])
def upload():

    if request.method == 'POST':
	
        img = request.files["my_image"]
        if img.filename=="":
            message='No file selected for uploading'
            return render_template("index.html", message= message)

        if img:
            img_path = "static/" + img.filename	
            img.save(img_path)
    
            pred= predict(img_path)

            for i in range(3):
                if pred[i] == 1.:
                    break
                prediction = classes[i]
        

            return render_template("index.html",img_path= img_path,  text=prediction)


if __name__ == "__main__":
    app.run(debug= True)
  