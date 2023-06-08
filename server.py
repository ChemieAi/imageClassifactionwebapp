from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import os

# Create app
app = Flask(__name__)

def load_keras_model():
    """Load in the pre-trained model"""
    global model
    LOCATE_PY_DIRECTORY_PATH = os.path.abspath(os.path.dirname(__file__))
    model = load_model(LOCATE_PY_DIRECTORY_PATH + '/model_weights.h5')
    # Required for model to work
    global graph
    graph = tf.compat.v1.get_default_graph()


# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Create form
    results={
        0:'AEROPLANE',
        1:'AUTOMOBILE',
        2:'BIRD',
        3:'CAT',
        4:'DEER',
        5:'DOG',
        6:'FROG',
        7:'HORSE',
        8:'SHIP',
        9:'TRUCK'
    }
    # On form entry and all conditions met
    # Extract information
    img = request.files['file']
    image = Image.open(img)
    image = image.resize((32,32))
    image = np.expand_dims(image,axis = 0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    print(results[pred])
    return jsonify(data=results[pred])

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_keras_model()
    # Run app
    app.run(host="0.0.0.0", port=80)