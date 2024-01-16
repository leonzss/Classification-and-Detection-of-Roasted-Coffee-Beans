from flask import Flask, render_template, request, Response, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from roboflow import Roboflow
from PIL import Image
import numpy as np
import os
import cv2
import atexit

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('my_model_expection.h5')
dic = {0: 'Dark', 1: 'Light', 2: 'Medium', 3: 'Unknown'}

camera = None
camera_running = False

def start_camera():
    global camera_running, camera
    if not camera_running:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            stop_camera()
            return jsonify({'status': 'Failed to open camera'})

        camera_running = True
        return jsonify({'status': 'Camera started'})

def stop_camera():
    global camera_running, camera
    if camera_running:
        camera.release()
        camera_running = False
    pass

# Daftarkan fungsi stop_camera() untuk dijalankan saat program berakhir
atexit.register(stop_camera)
        
def stop_camera_if_running():
    global camera_running
    if camera_running:
        stop_camera()
        return jsonify({'status': 'Camera stopped'})

def resize_frame(frame, width=224, height=224):
    return cv2.resize(frame, (width, height))

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = predictions[0][predicted_class[0]] * 100  # Confidence in percentage
    return dic.get(predicted_class[0], 'Unknown'), confidence

def predict_label_from_frame():
    global camera_running, camera
    if camera_running:
        success, frame = camera.read()
        if success:
            img = Image.fromarray(frame)
            img_path = "temp.jpg"
            img.save(img_path)
            return predict_label(img_path)

def generate_frames():
    global camera_running, camera
    while camera_running:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = resize_frame(frame, width=224, height=224)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    stop_camera_if_running()
    return render_template("dashboard.html")

@app.route("/cnn", methods=['GET', 'POST'])
def cnn():
    stop_camera_if_running()
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    stop_camera_if_running()
    return render_template("classification.html")

@app.route('/submit', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['my_image']
        
        # Pastikan folder uploads sudah ada, jika belum, buat folder tersebut
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)
        predicted_class, confidence = predict_label(img_path)

    return render_template("classification.html", predicted_class=predicted_class, confidence=confidence, img_path=img_path)

@app.route('/detection')
def detection():
    stop_camera_if_running()
    return render_template('detection.html')

@app.route('/start_camera')
def start_camera_route():
    start_camera()
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera')
def stop_camera_route():
    stop_camera()
    return jsonify({'status': 'Camera stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    label = predict_label_from_frame()
    return jsonify({'prediction': label})

if __name__ == '__main__':
    try:
        app.run(debug=False)
    finally:
        stop_camera()