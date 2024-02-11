# from flask import Blueprint, render_template, redirect, url_for
# from flask_admin import Admin
# from flask_admin.contrib.sqla import ModelView
# # from app import app, db, DetectionResult

# admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# # Inisialisasi objek Flask-Admin
# admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')

# # Tambahkan view untuk model DetectionResult
# admin.add_view(ModelView(DetectionResult, db.session))

# @admin_bp.route('/')
# def admin_home():
#     return render_template('admin/home.html')

# @admin_bp.route('/detection_results')
# def show_detection_results():
#     results = DetectionResult.query.all()
#     return render_template('admin/detection_results.html', results=results)

# @admin_bp.route('/delete_result/<int:result_id>')
# def delete_result(result_id):
#     result = DetectionResult.query.get_or_404(result_id)
#     db.session.delete(result)
#     db.session.commit()
#     return redirect(url_for('admin.show_detection_results'))

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