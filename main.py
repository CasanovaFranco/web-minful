
from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import dlib
import base64
import gdown

app = Flask(__name__, template_folder=os.path.abspath(''))

# Descargar el modelo previamente entrenado desde Google Drive
model_url = 'https://drive.google.com/uc?export=view&id=1LXiVwMHtRLkV8ZegBCcz1KDE4xlfIK8t'
model_path = 'asd_best_model.h5'
gdown.download(model_url, model_path, quiet=False)

# Descargar el detector de landmarks de dlib desde Google Drive
landmarks_url = 'https://drive.google.com/uc?export=view&id=1bnE0fryskX4ebGVG7yXi5MTIK9oFevN9'
landmarks_path = 'shape_predictor_68_face_landmarks.dat'
gdown.download(landmarks_url, landmarks_path, quiet=False)

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model(model_path)

# Cargar el detector de landmarks de dlib
predictor = dlib.shape_predictor(landmarks_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        predicted_text = "Autismo Detectado"
    else:
        predicted_text = "No tiene autismo"

    # Detectar landmarks en la imagen de salida
    output_img = np.array(image)
    output_img_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    faces = detector(output_img_gray)

    for face in faces:
        landmarks = predictor(output_img_gray, face)
        for n in range(36, 48):  # Región de los ojos
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)  # Dibujar círculos verdes

    # Convertir imagen de salida a base64
    _, buffer = cv2.imencode('.jpg', output_img)
    output_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'predicted_text': predicted_text, 'output_img_base64': output_img_base64})

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    app.run(debug=True)