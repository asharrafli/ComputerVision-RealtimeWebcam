from flask import Flask, render_template, Response, send_file, request
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

current_effect = 'original' # default effect
current_layout = 'single'
capture_image = None
last_frame = None  # Variabel global untuk menyimpan frame terakhir

# Buat objek kamera global agar webcam tidak mati-nyala
camera = cv2.VideoCapture(0)

import atexit
atexit.register(lambda: camera.release())

# original effect
def original(frame):
    return frame

# sketch effect 
def sketch(frame):
    # mengubah gambar ke grayscale terlebih dahulu
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # membersihkan noise dengan Gaussian blur
    frame_blur= cv2.GaussianBlur(frame_gray,(5,5),0)
    # emplementasi deteksi tepi dengan Canny
    edges = cv2.Canny(frame_blur,30,60)
    # do an inverse binary thresholding
    ret, mask = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY_INV)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# pencil effect
def pencil_effect(frame):
    # mengubah gambar ke grayscale terlebih dahulu
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mendeteksi tepi dengan Canny
    edges = cv2.Laplacian(frame_gray, cv2.CV_8U, ksize=5)
    # do an inverse binary thresholding
    ret, edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

#sepia effect 
def sepia(frame): 
    # membuat matriks filter sepia
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    # menerapkan filter sepia
    sepia_frame = cv2.transform(frame, sepia_filter)
    # memastikan nilai piksel berada dalam rentang yang valid
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    return sepia_frame.astype(np.uint8)


#mengaplikasikan efek yang akan dipilih
def apply_effect(frame,effect_name):
    if effect_name == 'sketch':
        return sketch(frame)
    elif effect_name == 'pencil':
        return pencil_effect(frame)
    elif effect_name == 'sepia':
        return sepia(frame)
    return original(frame)

# fungsi untuk menampilkan kamera

def generate_camera():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        # mengubah ukuran frame
        frame = cv2.resize(frame, (640, 480))
        last_frame = frame.copy()  # Simpan frame terakhir

        # logika pada saat frame diambil dan efek diterapkan
        if current_layout == '1frame':
            proc_frame = apply_effect(frame.copy(), current_effect)
            ret, buffer = cv2.imencode('.jpg', proc_frame)
            frame_bytes = buffer.tobytes()
        else: #jika 4 frame
            frame1 = apply_effect(frame.copy(), current_effect)
            frame2 = apply_effect(frame.copy(), current_effect)
            frame3 = apply_effect(frame.copy(), current_effect)
            frame4 = apply_effect(frame.copy(), current_effect)

            # menggabungkan frame menjadi satu
            top_row = np.hstack((frame1, frame2))
            bottom_row = np.hstack((frame3, frame4))
            combined_frame = np.vstack((top_row, bottom_row))

            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_layout')
def toggle_layout():
    global current_layout
    # Toggle antara 1frame dan 4frame
    if current_layout == '1frame':
        current_layout = '4frame'
    else:
        current_layout = '1frame'
    return f"Layout changed to {current_layout}"

@app.route('/set_effect/<effect>')
def set_effect(effect):
    global current_effect
    current_effect = effect
    return f"Effect changed to {current_effect}"

@app.route('/capture')
def capture():
    global capture_image, last_frame
    # Ambil frame terakhir dari streaming, bukan dari camera.read()
    if last_frame is not None:
        frame = last_frame.copy()
        if current_layout == '1frame':
            capture_image = apply_effect(frame.copy(), current_effect)
        else:
            frame1 = apply_effect(frame.copy(), current_effect)
            frame2 = apply_effect(frame.copy(), current_effect)
            frame3 = apply_effect(frame.copy(), current_effect)
            frame4 = apply_effect(frame.copy(), current_effect)
            top_row = np.hstack((frame1, frame2))
            bottom_row = np.hstack((frame3, frame4))
            capture_image = np.vstack((top_row, bottom_row))
        return 'Captured successfully!'
    return 'Failed to capture image.'

# route untuk mendownload
@app.route('/download')
def download():
    global capture_image
    if capture_image is not None:
        _, buffer = cv2.imencode('.jpg', capture_image)
        mem_file = BytesIO(buffer)
        mem_file.seek(0)
        filename = f"captured_image_{current_effect}.jpg"
        return send_file(mem_file, mimetype='image/jpeg',
                         as_attachment=True, download_name=filename)
    return 'No image captured.'

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
    # app.run(debug=True, port=5000) # jika ingin menggunakan localhost saja