from playsound import playsound
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import sys
sys.path.append(r'C:\Users\STUDYSPC\Desktop\mask_detect_system\yolov5')
sys.path.append(r'C:\Users\STUDYSPC\Desktop\mask_detect_system\yolov5\utils')
import torch
import cv2
import time
import numpy as np
import face_recognition
import os
from twilio.rest import Client
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Twilio bilgileri
account_sid = "AC15360d3b6cced285800e3ad87f182a52"
auth_token = "29798552db72cb1e13d811ffd6c8b171"
twilio_phone_number = "+12177658138"
target_phone_number = "+905383137850"
client = Client(account_sid, auth_token)

def play_alert_sound():
    playsound("UYARI.mp3")

def play_wrong_sound():
    playsound("UYARI2.mp3")

def send_sms(message):
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=target_phone_number
    )

# Kayıtlı yüzler\
known_face_encodings = []
known_face_names = []
face_dir = "known_faces"
for filename in os.listdir(face_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(face_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

def letterbox(im, new_shape=(512,512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

last_sms_time = 0
last_sound_time = 0
sms_interval = 10
sound_interval = 11
program_start_time = time.time()  # Kamera açıldığı zamanı tut
device = select_device('cpu')
weights_path = r'C:\Users\STUDYSPC\Desktop\mask_detect_system\yolov5\runs\train\exp\weights\best.pt'
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (512, 512)
cap = cv2.VideoCapture(0)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    im = letterbox(frame, imgsz, stride=stride)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im).to(device)
    im_tensor = im_tensor.float() / 255.0
    if im_tensor.ndimension() == 3:
        im_tensor = im_tensor.unsqueeze(0)
    pred = model(im_tensor)
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]
    pred[:, :4] = scale_boxes(im_tensor.shape[2:], pred[:, :4], frame.shape).round()

    yuz = []
    maske = []
    for *xyxy, conf, cls in pred:
        label = names[int(cls)]
        box = [int(x.item()) for x in xyxy]
        if label == "yuz":
            yuz.append(box)
        elif label == "maske":
            maske.append(box)

    for mask_box in maske:
        x1, y1, x2, y2 = mask_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Maske", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for face_box in yuz:
        x1, y1, x2, y2 = face_box
        face_image = frame[y1:y2, x1:x2]
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)
        person_name = "Bilinmeyen çalışan"
        if face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                person_name = known_face_names[best_match_index]

        best_iou = 0
        best_mask = None
        for mask_box in maske:
            current_iou = iou(face_box, mask_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_mask = mask_box

        if best_iou > 0.4:
            color = (0, 255, 0)
            label = "Maske Takili"
        elif 0.05 < best_iou <= 0.4:
            color = (0, 255, 255)
            label = "Maske Yanlis Takilmis!!"
            current_time = time.time()
            if current_time - last_sound_time > sound_interval:
                play_wrong_sound()
                last_sound_time = current_time
        elif len(maske) > 0:
            color = (0, 165, 255)
            label = "Maske Var Ama Takili Degil!!"
            current_time = time.time()
            if current_time - program_start_time > 10:
                if current_time - last_sms_time > sms_interval:
                    send_sms(f"⚠️ Uyarı: {person_name} maske takmıyor!")
                    last_sms_time = current_time
                if current_time - last_sound_time > sound_interval:
                    play_alert_sound()
                    last_sound_time = current_time
        else:
            color = (0, 0, 255)
            label = "Maske Yok"
            current_time = time.time()
            if current_time - program_start_time > 10:
                if current_time - last_sms_time > sms_interval:
                    send_sms(f"⚠️ Uyarı: {person_name} maske takmıyor!")
                    last_sms_time = current_time
                if current_time - last_sound_time > sound_interval:
                    play_alert_sound()
                    last_sound_time = current_time
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Maske Kontrol Sistemi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
