from flask import Flask, render_template, Response
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

current_effect = 'original' # default effect
current_layout = 'single'
capture_image = None


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
    cap = cv2.VideoCapture(0) #untuk membuka kamera di webcam default
    while True:
        success, frame = cap.read()
        if not success:
            break
        # mengubah ukuran frame
        frame = cv2.resize(frame, (640, 480))

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
        
        cap.release()

@app.route('/capture')
def capture():
    
