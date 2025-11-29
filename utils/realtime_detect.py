import torch
import cv2
from playsound import playsound
import threading
import sys
sys.path.append('./yolov5')


# Modeli yükle
from yolov5.models.common import DetectMultiBackend
import torch

model = DetectMultiBackend('best.pt', device='cpu')
model.conf = 0.4  # minimum güven eşiği

# Sesli uyarı fonksiyonu
def play_alert():
    playsound("alert.mp3")

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = results.pandas().xyxy[0]['name'].tolist()

    if 'mask' not in labels:  # Maske takılmamışsa uyarı ver
        cv2.putText(frame, "MASKESIZ!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        threading.Thread(target=play_alert).start()

    # Çıktıyı göster
    cv2.imshow("Gerçek Zamanlı Maske Tespiti", results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
