import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import current_app
from sqlalchemy import text
import cv2 #1
import numpy as np
import dlib #1
import os
import pytz #1
import requests #1
from imutils import face_utils #1
import time
import asyncio
from pygame import mixer #1
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///driver_detection.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

ESP32_CAM_IP = '192.168.192.212'
ESP32_CAM_PORT = 80

jakarta_timezone = pytz.timezone('Asia/Jakarta')

mixer.init()
no_driver_sound = mixer.Sound('nodriver_audio.wav')
sleep_sound = mixer.Sound('sleep_sound.wav')
tired_sound = mixer.Sound('rest_audio.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

nb_classifier = GaussianNB()
scaler = StandardScaler()

classifier_trained = False

result_label = ""

class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_type = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255))
    result = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime(timezone=True), server_default=text('now()'))

    def __repr__(self):
        return f'<DetectionResult {self.result_type} at {self.timestamp}>'


def train_classifier():
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_train = ['active', 'sleep', 'active', 'sleep']
    X_train_scaled = scaler.fit_transform(X_train)
    nb_classifier.fit(X_train_scaled, y_train)


def extract_features(landmarks):
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            distance = compute(landmarks[i], landmarks[j])
            features.append(distance)
            if len(features) == 2:
                return features
    return features


def classify_frame(frame):
    if frame is None:
        print("Error: Gagal mengambil citra gambar dari frame")
        return 'active'

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if faces:
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            features = extract_features(landmarks)
            features_scaled = scaler.transform([features])
            prediction = nb_classifier.predict(features_scaled)
            return prediction[0]
    return 'active'


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.22:
        return 'active'
    else:
        return 'sleep'


def mouth_aspect_ratio(mouth):
    A = compute(mouth[2], mouth[10])
    B = compute(mouth[4], mouth[8])
    C = compute(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


(mStart, mEnd) = (49, 68)


async def tired():
    start = time.time()
    rest_time_start = start
    tired_sound.play()
    a = 0
    while (time.time()-start < 9):
        if(time.time()-rest_time_start > 3):
            tired_sound.play()
    tired_sound.stop()
    return


def detech():
    global result_label, ESP32_CAM_IP, ESP32_CAM_PORT
    result = None
    with app.app_context():
        train_classifier()
        sleep_sound_flag = 0
        no_driver_sound_flag = 0
        sleep_sound_stop_time = 0
        capture_count = 0
        capture_folder = os.path.join(current_app.root_path, 'captures')
        os.makedirs(capture_folder, exist_ok=True)
        yawning = 0
        no_yawn = 0
        sleep = 0
        active = 0
        status = ""
        color = (0, 0, 0)
        no_driver = 0
        frame_color = (0, 255, 0)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cap.read()

        time.sleep(1)
        start = time.time()
        no_driver_time = time.time()
        no_driver_sound_start = time.time()

    capture_path = None
    db.create_all()

    while True:
        if frame is None:
            break
        _, frame = cap.read()
        if not _:
            print("Error reading frame from ESP32-CAM.")
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print("Warning: Invalid color channels in the frame.")
            continue
        face_frame = frame.copy()
        faces = detector(gray, 0)

        if faces:
            no_driver_sound_flag = 0
            no_driver_sound.stop()
            no_driver = 0
            no_driver_time = time.time()

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_blink = blinked(landmarks[36], landmarks[37],
                                     landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43],
                                      landmarks[44], landmarks[47], landmarks[46], landmarks[45])
                mouth = landmarks[mStart:mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR

                if (mar > 0.70):
                    sleep = 0
                    active = 0
                    yawning += 1
                    status = "Menguap"
                    color = (255, 0, 0)
                    frame_color = (255, 0, 0)
                    sleep_sound_flag = 0
                    sleep_sound.stop()
                elif (left_blink == 'sleep' or right_blink == 'sleep'):
                    if (yawning > 20):
                        no_yawn += 1
                    sleep += 1
                    yawning = 0
                    active = 0
                    if (sleep > 5):
                        status = "Tertidur!"
                        color = (0, 0, 255)
                        frame_color = (0, 0, 255)
                        if sleep_sound_flag == 0:
                            sleep_sound.play()
                        sleep_sound_flag = 1

                        if time.time() - sleep_sound_stop_time > 5:
                            sleep_sound_stop_time = time.time()
                            result = classify_frame(frame)
                            result_label = result
                            if result:
                                detection_result = DetectionResult(result_type=result)
                                capture_filename = f"capture_{capture_count}.jpg"
                                capture_path = os.path.join(capture_folder, capture_filename)
                                cv2.imwrite(capture_path, frame)
                                detection_result.image_path = capture_path
                                db.session.add(detection_result)
                                db.session.commit()
                                capture_count += 1
                else:
                    if (yawning > 20):
                        no_yawn += 1
                    yawning = 0
                    sleep = 0
                    active += 1
                    status = "Aman"
                    color = (0, 255, 0)
                    frame_color = (0, 255, 0)
                    if active > 5:
                        sleep_sound_flag = 0
                        sleep_sound.stop()

                cv2.putText(frame, status, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                if (time.time()-start < 60 and no_yawn >= 3):
                    no_yawn = 0
                    asyncio.run(tired())
                elif time.time()-start > 60:
                    start = time.time()

                for n in range(0, 68):
                    (x, y) = landmarks[n]
                    cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

            else:
                no_driver += 1
                sleep_sound_flag = 0
                sleep_sound.stop()
                if(no_driver > 10):
                    status = "Tidak ada pengemudi"
                    color = (0, 0, 0)
                if time.time()-no_driver_time > 5:
                    if(no_driver_sound_flag == 0):
                        no_driver_sound.play()
                        no_driver_sound_start = time.time()
                    else:
                        if(time.time()-no_driver_sound_start > 3):
                            no_driver_sound.play()
                            no_driver_sound_start = time.time()
                    no_driver_sound_flag = 1
            cv2.putText(frame, status, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.imshow("PENGEMUDI", frame)
            cv2.imshow("LANDMARKS WAJAH", face_frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            
    if result_label == 'sleep':
        send_signal_to_esp32('activate')
    else:
        send_signal_to_esp32('deactivate')
    detection_result = DetectionResult(result=result)
    
    
    no_driver_sound.stop()
    sleep_sound.stop()
    tired_sound.stop()
    cap.release()
    cv2.destroyAllWindows()

    result = classify_frame(frame)
    result_label = result
    capture_filename = f"capture_{capture_count}.jpg"
    capture_path = os.path.join(capture_folder, capture_filename)
    cv2.imwrite(capture_path, frame)
    detection_result.image_path = capture_path
    db.session.add(detection_result)
    db.session.commit()

    return render_template("index.html", result=result_label)


def send_signal_to_esp32(result):
    if result == 'sleep':
        requests.get(f'http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/activate_sensors')


@app.route("/open_camera")
def open():
    detech()
    timeout = 60
    data = request.get_json()
    frame_data = data["frame"]
    frame = np.array(frame_data, dtype=np.uint8).reshape(480, 640, 3)
    result = classify_frame(frame)
    result_label = result
    if result_label == 'sleep':
        control_sensors('activate')
    else:
        control_sensors('deactivate')
    detech(frame)
    return redirect, jsonify, Response("/", {"classification": result_label}, open_camera(), stream_with_context(detech()), mimetype='multipart/x-mixed-replace; boundary=frame', content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True, timeout=timeout)


@app.route('/control_sensors', methods=['GET'])
def control_sensors():
    action = request.args.get('action')
    
    control_sensors(action)
    
    return "Sensor telah dikontrol."





@app.route("/save_image", methods=["POST"])
def save_image():
    # ambil data gambar dari permintaan POST
    image_data = request.form.get('image')

    # simpan data gambar ke db
    save_image_to_db(image_data)
    return 'Gambar berhasil disimpan.'

def save_image_to_db(image_data):
    image_path = save_image_to_disk(image_data)
    new_detection = DetectionResult(result_type='Tertidur', image_path=image_path)
    db.session.add(new_detection)
    db.session.commit()

def save_image_to_disk(image_data):
    path = f'image_{datetime.gmtnow().strftime("%Y%m%d%H%M%S")}.jpg'
    with open(path, 'wb') as file:
        file.write(base64.b64decode(image_data))

    return path

@app.route("/train_classifier")
def train():
    train_classifier()
    return "Klasifikasi model sukses."

@app.route("/classify_frame", methods=["POST"])
def classify():
    data = request.get_json()
    frame_data = data["frame"]
    frame = np.array(frame_data, dtype=np.uint8).reshape(480, 640, 3)
    result = classify_frame(frame)
    update_classification_result(result)
    return jsonify({"classification_result": result})

@app.route("/get_naive_bayes_result")
def get_naive_bayes_result():
    global result_label
    return jsonify({"result": result_label})


@app.route("/detection_history")
def detection_history():
    history = DetectionResult.query.all()
    return render_template("detection_history.html", history=history)

@app.route("/count_detection", methods=["POST"])
def count_detections():
    with app.app_context():
        try:
            result_type = request.form['result_type']
            new_detection = DetectionResult(result_type=result_type)
            db.session.add(new_detection)
            db.session.commit()
            # global result_label
            # result_label = new_detection.result_type
        # except KeyError as e:
        #     return jsonify({'error': f"Error: {e}"}), 400

            sleep_count = DetectionResult.query.filter_by(result_type='Tertidur').count()
            yawning_count = DetectionResult.query.filter_by(result_type='Menguap').count()
            active_count = DetectionResult.query.filter_by(result_type='Aman').count()
    
        # Kirim response JSON yang berisi data deteksi aktual
        # Panggil fungsi JavaScript untuk meng-update nilai-nilai di frontend
            return f"<script>updateCounts({sleep_count}, {yawning_count}, {active_count});</script>"
        except KeyError as e:
            return jsonify({'error': f"Error: {e}"}), 400

    # return render_template("index.html", sleep_count=sleep_count, yawning_count=yawning_count, active_count=active_count)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    # mendapatkan hasil deteksi dari model
    detech()
    return jsonify({'result': result_label})
    # menyampaikan signal ke ESP


@app.route("/")
def home():
    with app.app_context():
        sleep_count = DetectionResult.query.filter_by(result_type='Tertidur').count()
        yawning_count = DetectionResult.query.filter_by(result_type='Menguap').count()
        active_count = DetectionResult.query.filter_by(result_type='Aman').count()
        global result_label
    return render_template("index.html", result=result_label, sleep_count=sleep_count, yawning_count=yawning_count, active_count=active_count)

if __name__ == "__main__":
    with app.app_context():
        result = DetectionResult.query.all()
        db.create_all()
    train_classifier()
    app.run(host='0.0.0.0', port=5000, debug=True)