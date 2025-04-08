from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import pandas as pd
from datetime import datetime
import pytz
import threading
import asyncio
import telegram
import os

app = Flask(__name__)

# Đường dẫn thư mục uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Variables
cap = None
is_camera_on = False
video_paused = False
frame_count = 0
alert_telegram_each = 5
frame_skip_threshold = 5  # Thay 3 thành 5 khi chạy trên RENDER
area = []
model = YOLO('best.pt')
class_list = []
last_alert = None
selected_class = "All"
telegram_token = ""
telegram_chat_id = ""
show_warning = False  # Biến để bật/tắt cảnh báo
draw_area_enabled = False  # Biến để bật/tắt chế độ vẽ vùng

# Function to read coco.txt
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Load class list
class_list = read_classes_from_file('coco.txt')

# Function to start the webcam feed
def start_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)  # Use default webcam
        is_camera_on = True
        video_paused = False

# Function to stop the webcam feed
def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False

# Function to pause or resume the video
def pause_resume_video():
    global video_paused
    video_paused = not video_paused

# Function to select a video file
def select_file(file_path):
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()
    cap = cv2.VideoCapture(file_path)
    is_camera_on = True
    video_paused = False

# Function to reset points
def reset_app():
    global area
    area.clear()

# Async function to send photo via Telegram
async def send_photo_async(bot, chat_id, photo_path):
    async with bot:
        await bot.send_photo(chat_id=chat_id, photo=open(photo_path, "rb"),
                             caption="Phát hiện người xâm nhập !!!")

# Sync function for threading
def send_telegram_sync():
    global telegram_token, telegram_chat_id
    photo_path = "alert.png"
    try:
        print("Sending Telegram alert...")
        bot = telegram.Bot(token=telegram_token)
        asyncio.run(send_photo_async(bot, telegram_chat_id, photo_path))
        print("Gửi thành công")
    except Exception as ex:
        print("Không thể gửi tin nhắn tới Telegram ", ex)

# Warning function
def warning(image):
    global last_alert, show_warning
    if show_warning:  # Chỉ vẽ cảnh báo nếu show_warning là True
        cv2.putText(image, "CANH BAO CO NGUOI XAM NHAP!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Kiểm tra thời gian gửi Telegram
        if (last_alert is None) or ((datetime.now(pytz.utc) - last_alert).total_seconds() > alert_telegram_each):
            last_alert = datetime.now(pytz.utc)
            cv2.imwrite("alert.png", cv2.resize(image, dsize=None, fx=0.5, fy=0.5))
            threading.Thread(target=send_telegram_sync).start()
    return image

# Function to generate video frames
def generate_frames():
    global is_camera_on, video_paused, frame_count, selected_class
    while True:
        if not is_camera_on or video_paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # Nếu video kết thúc, lặp lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % frame_skip_threshold != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # TRÊN RENDER
        frame = cv2.resize(frame, (640, 360))

        # TRÊN MÁY
        #frame = cv2.resize(frame, (1020, 500))


        # YOLO detection
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        person_detected = False  # Biến để kiểm tra xem có phát hiện person không

        # Vẽ vùng area nếu đã có các điểm
        if area:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (209, 21, 102), 2)

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            # Kiểm tra nếu class là "person" hoặc selected_class là "All"
            if (c == "person" and (selected_class == "All" or selected_class == "person")):
                person_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung xanh cho person
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                if area and len(area) >= 3:  # Đảm bảo có ít nhất 3 điểm để tạo vùng
                    mid_x = (x1 + x2) // 2
                    mid_y = y2
                    result = cv2.pointPolygonTest(np.array(area), (mid_x, mid_y), False)
                    if result >= 0:  # Nếu person nằm trong vùng
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vẽ khung đỏ
                        cv2.circle(frame, (mid_x, mid_y), 4, (255, 0, 0), -1)
                        frame = warning(image=frame)  # Gọi hàm warning
                        print("Person detected in area!")  # Debug

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html', classes=class_list)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    start_webcam()
    return "Started"

@app.route('/stop', methods=['POST'])
def stop():
    stop_webcam()
    return "Stopped"

@app.route('/pause_resume', methods=['POST'])
def pause_resume():
    pause_resume_video()
    return "Paused/Resumed"

@app.route('/reset', methods=['POST'])
def reset():
    reset_app()
    return "Points reset"

@app.route('/set_area', methods=['POST'])
def set_area():
    global area, draw_area_enabled
    if draw_area_enabled:  # Chỉ thêm điểm nếu chế độ vẽ được bật
        # Xử lý tọa độ dạng float trước khi ép kiểu sang int
        x = int(float(request.form['x']))
        y = int(float(request.form['y']))
        area.append((x, y))
        print(f"Added point: ({x}, {y})")  # Debug
        print(f"Current area: {area}")  # Debug
    return "Point added"

@app.route('/set_class', methods=['POST'])
def set_class():
    global selected_class
    selected_class = request.form['class']
    return "Class updated"

@app.route('/set_telegram', methods=['POST'])
def set_telegram():
    global telegram_token, telegram_chat_id
    telegram_token = request.form['token']
    telegram_chat_id = request.form['chat_id']
    return "Telegram settings updated"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        select_file(file_path)
        return "File uploaded and video started", 200

@app.route('/toggle_warning', methods=['POST'])
def toggle_warning():
    global show_warning
    show_warning = not show_warning
    print(f"Show warning: {show_warning}")  # Debug
    return "Warning toggled"

@app.route('/toggle_draw_area', methods=['POST'])
def toggle_draw_area():
    global draw_area_enabled
    draw_area_enabled = not draw_area_enabled
    print(f"Draw area enabled: {draw_area_enabled}")  # Debug
    return "Draw area toggled"

# PORT CHẠY TRÊN MÁY
# if __name__ == '__main__':
#     app.run(debug=True)


# PORT CHẠY TRÊN RENDER
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Lấy port từ biến môi trường
    app.run(host='0.0.0.0', port=port, debug=False)  # Chạy trên host 0.0.0.0