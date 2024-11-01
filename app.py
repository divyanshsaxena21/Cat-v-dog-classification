from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('my_dogs_vs_cats_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize
    img = img.reshape((1, 256, 256, 3))

    prediction = model.predict(img)
    label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
