
import base64
from datetime import datetime
from flask import Flask, Response,render_template,redirect,jsonify, request, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import current_app
from sqlalchemy import text
import cv2
import numpy as np
import dlib
import os
import pytz
import requests
from imutils import face_utils
import time
import asyncio
import http.client
import urllib3
from urllib3.util.retry import Retry
from io import BytesIO
from pygame import mixer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///driver_detection.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

ESP32_CAM_IP = '192.168.192.212'

jakarta_timezone = pytz.timezone('Asia/Jakarta')



# frame_data = response.data
mixer.init()
no_driver_sound = mixer.Sound('nodriver_audio.wav')
sleep_sound = mixer.Sound('sleep_sound.wav')
tired_sound = mixer.Sound('rest_audio.wav')

# inisialisasi deteksi wajah & lekungan
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

nb_classifier = GaussianNB()
scaler = StandardScaler()

classifier_trained = False

result_label = ""



# buat kelas untuk hasil deteksi yang disimpan pada database
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_type = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255)) ## kolom untuk menyimpan image path
    result = db.Column(db.String(50)) ## kolom simpan hasil deteksi
    timestamp = db.Column(db.DateTime(timezone =True), server_default=text('now()'))

    # db.create_all()


    def __repr__(self):
        return f'<DetectionResult {self.result_type} at {self.timestamp}>'

def get_frame_from_esp32_cam():
    try:
        response = requests.get(f'http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/open_camera')
        frame_data = response.content
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        # frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    except Exception as e:
        print(f"Error reading frame from ESP32-CAM: {e}")
        return None


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
    ratio = up/(2.0*down)

    # Checking if it is blinked
    if (ratio > 0.22):
        return 'active'
    else:
        return 'sleep'


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = compute(mouth[2], mouth[10])  # 51, 59
    B = compute(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = compute(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


(mStart, mEnd) = (49, 68)


async def tired():
    start = time.time()
    rest_time_start=start
    tired_sound.play()
    a = 0
    while (time.time()-start < 9):
        if(time.time()-rest_time_start>3):
            tired_sound.play()
        # cv2.imshow("USER",tired_img)
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
            capture_count = 0 # simpan jumlah frame yang dicapture
            capture_folder = os.path.join(current_app.root_path, 'captures') ## folder untuk simpan foto
            os.makedirs(capture_folder, exist_ok=True)
            yawning = 0
            no_yawn = 0
            sleep = 0
            active = 0
            status = ""
            color = (0, 0, 0)
            no_driver=0
            frame_color = (0, 255, 0)

          
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # cap = cv2.VideoCapture(ESP32_CAM_URL)
            ret, frame = cap.read()

            time.sleep(1)
            start = time.time()
            no_driver_time=time.time()
            no_driver_sound_start = time.time()


    # inisialisasi variabel capture_path di luar looping
    capture_path = None

    while True:
        # frame = cv2.cap()
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


        # detected face in faces array
        if faces:
         no_driver_sound_flag=0   
         no_driver_sound.stop()   
         no_driver=0  
         no_driver_time=time.time() 
        #  sleep_sound.stop()
         for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)
            # cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            mouth = landmarks[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR

            # Now judge what to do for the eye blinks

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

                    # bagian yang ditambahkan
                    if time.time() - sleep_sound_stop_time > 5:
                        sleep_sound_stop_time = time.time()

                        # simpan hasil deteksi tidur
                        result = classify_frame(frame)
                        result_label = result
                        if result:
                            detection_result = DetectionResult(result_type=result)

                            # simpan foto ke folder
                            capture_filename = f"capture_{capture_count}.jpg"
                            capture_path = os.path.join(capture_folder, capture_filename)
                            cv2.imwrite(capture_path, frame)

                            # update path foto
                            detection_result.image_path = capture_path

                            db.session.add(detection_result)
                            db.session.commit()
                            capture_count += 1
                        # if result_label == 'sleep':
                        #     detection_result = DetectionResult(result=result, image_path=capture_path)

                        # ## simpan foto keq folder
                        #     capture_filename = f"capture_{capture_count}.jpg"
                        #     capture_path = os.path.join(capture_folder, capture_filename)
                        #     cv2.imwrite(capture_path, frame)

                        # # update path foto
                        #     detection_result.image_path = capture_path

                        #     db.session.add(detection_result)
                        #     db.session.commit()
                        #     capture_count += 1
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
            no_driver+=1
            sleep_sound_flag = 0
            sleep_sound.stop()
            if(no_driver>10):
              status="Tidak ada pengemudi"
              color=(0,0,0)
            if time.time()-no_driver_time>5:
                if(no_driver_sound_flag==0):
                   no_driver_sound.play()
                   no_driver_sound_start=time.time()
                else:
                    if(time.time()-no_driver_sound_start>3):
                        no_driver_sound.play()
                        no_driver_sound_start=time.time()
                no_driver_sound_flag=1
        cv2.putText(frame, status, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("PENGEMUDI", frame)
        cv2.imshow("LANDMARKS WAJAH", face_frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
  

    no_driver_sound.stop()
    sleep_sound.stop()
    tired_sound.stop()
    cap.release()
    cv2.destroyAllWindows()   

    result = classify_frame(frame)
    result_label = result

    ## simpan hasil deteksi ke database
    detection_result = DetectionResult(result=result)

    # simpan foto ke folder
    capture_filename = f"capture_{capture_count}.jpg"
    capture_path = os.path.join(capture_folder, capture_filename)
    cv2.imwrite(capture_path, frame)

    # update path foto
    detection_result.image_path = capture_path

    db.session.add(detection_result)
    db.session.commit()

    return render_template("index.html", result=result_label)


def control_sensor(action):
    try:
        esp32_cam_url = f"http://{ESP32_CAM_IP}:80/control_sensor?action={action}"
        response = requests.get(esp32_cam_url)
        result = response.text
        return f"Kontrol Sensor: {result}"
    except Exception as e:
        return f"Error: {e}"
        

@app.route("/open_camera")
def open():
    
    detech()
    # frame = detech()
    timeout = 60
    # ret, frame = requests.get(ESP_CAM_URL)
    if frame is None or frame.empty():
        return "Error: Gagal mengambil citra gambar dari frame"
    _,frame = cv2.read()
    print("open camera")
    data = request.get_json()
    # response = requests.get(ESP_CAM_URL)    
    
    frame_data = data["frame"]


    frame = np.array(frame_data, dtype=np.uint8).reshape(480, 640, 3)
    result = classify_frame(frame)
    result_label = result
    if result_label == 'sleep':
        control_sensor('activate')
    else:
        control_sensor('deactivate')
    detech(frame) #meneruskan frame ke detech
    return redirect, jsonify, Response("/", {"classification": result_label}, open_camera(), mimetype='multipart/x-mixed-replace; boundary=frame', content_type='multipart/x-mixed-replace; boundary=frame', status=200, direct_passthrough=True, timeout=timeout)

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

@app.route("/capture_frame")

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
def send_signal_to_esp32(result):
    if result == 'sleep':
        requests.get(f'http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/activate_sensors')

# @app.route('/update_status', methods=['POST'])
# def update_status():
#     data = request.get_json()
#     status = data.get('status', 'unknown')
    
#     # Lakukan sesuatu dengan status yang diterima (misalnya, simpan ke database)
    
#     return 'Status updated successfully'

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
    app.run(debug=True)