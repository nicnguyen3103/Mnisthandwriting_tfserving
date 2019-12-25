import base64
import re
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
# pip install google-api-python-client
import googleapiclient.discovery

app = Flask(__name__)


project_id = "tensorflowdeployment" # change this to your project ID
model_id = "my_mnist_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

def parse_image(imgData):
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(imgstr)
    return img_decode

def preprocess_single_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = (255 - image) / 255.0  # normalize to [0,1] range
    image = tf.reshape(image, (1, 28, 28, 1))
    return image

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload/", methods=["POST"])
def upload_file():
    img_raw = parse_image(request.get_data())
    image = preprocess_single_image(img_raw)
    # change into numpy() array 
    image_np = image.numpy()
    # use python dict here 
    input_data_json = {"signature_name": "serving_default",
                       "instances": image_np.tolist()}
    request_ml = ml_resource.predict(name=model_path, body=input_data_json)
    response = request_ml.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    # dense_13 is the output layer of my model 
    y_proba = np.array([pred['dense_13'] for pred in response["predictions"]])
    prediction = np.argmax(y_proba, axis=1)
    return str(prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) 