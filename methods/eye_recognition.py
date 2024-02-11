import cv2
import numpy as np
import serial

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi detektor wajah dan mata
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inisialisasi koneksi Serial
ser = serial.Serial('COM3', 9600)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Konversi ke skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata di dalam wajah
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Jika mata tertutup (contoh: menggunakan threshold tinggi)
            if eh < 25:
                # Kirim sinyal kantuk ke Arduino
                ser.write(b'1')
            else:
                # Kirim sinyal tidak kantuk ke Arduino
                ser.write(b'0')

    # Tampilkan hasil
    cv2.imshow('Deteksi Kantuk', frame)

    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
