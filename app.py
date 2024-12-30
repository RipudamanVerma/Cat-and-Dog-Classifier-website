from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__, static_folder='static')

# Load the pre-trained model
model = tf.keras.models.load_model('cat_dog_classifier.h5')


def prepare_image(image):
    img = Image.open(image)
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = img.reshape((1, 150, 150, 3))
    return img


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img = prepare_image(file)

    # Predict the class
    prediction = model.predict(img)

    if prediction[0] > 0.5:
        result = 'Dog'
    else:
        result = 'Cat'

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
