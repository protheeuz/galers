from flask import Flask,render_template,redirect,jsonify, request
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import asyncio
from pygame import mixer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

mixer.init()
no_driver_sound = mixer.Sound('nodriver_audio.wav')
sleep_sound = mixer.Sound('sleep_sound.wav')
tired_sound = mixer.Sound('rest_audio.wav')

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

nb_classifier = GaussianNB()
scaler = StandardScaler()

classifier_trained = False

result_label = ""


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
    return features



def classify_frame(frame):
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
    # return 'unknown'


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
    global result_label
    # status marking for current state
    sleep_sound_flag = 0
    no_driver_sound_flag = 0
    yawning = 0
    no_yawn = 0
    sleep = 0
    active = 0
    status = ""
    color = (0, 0, 0)
    no_driver=0
    frame_color = (0, 255, 0)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

   
    time.sleep(1)
    start = time.time()
    no_driver_time=time.time()
    no_driver_sound_start = time.time()

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                # print("tired")
                # asyncio.run(put_image(frame))
                # time.sleep(2)
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
        cv2.imshow("68 POINTS", face_frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    no_driver_sound.stop()
    sleep_sound.stop()
    tired_sound.stop()
    cap.release()
    cv2.destroyAllWindows()   

    result = classify_frame(frame)
    result_label = result
    return render_template("index.html", result=result_label)

@app.route("/open_camera")
def open():
    detech()
    print("open camera")
    return redirect("/")

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

@app.route("/")
def home():
    global result_label
    return render_template("index.html", result=result_label)


if __name__ == "__main__":
    train_classifier()
    app.run(debug=True)