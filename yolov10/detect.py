import cv2
import torch
import imutils
import time
from ultralytics import YOLOv10

# YOLOv10 모델 로드 (ultralytics 라이브러리 사용)
model = YOLOv10('./solar_panel_maintenance/weights/best.pt')

# 카테고리 이름 설정
category_names = ['bird_drop', 'cracked', 'dusty', 'panel']
# 비디오 파일 경로
video_path = './solar_panel_fault_video.mp4'
camera = cv2.VideoCapture(video_path)

if not camera.isOpened():
    print("Error: Could not open video file.")
    exit()

# 루프 시작
while True:
    grabbed, frame = camera.read()

    if not grabbed:
        print("End of video file reached.")
        break

    frame = imutils.resize(frame, width=700)
    # frame = cv2.flip(frame, 1)

    # YOLOv10을 사용하여 프레임 예측
    results = model(frame)

    # 탐지된 객체 가져오기
    detections = results[0].boxes.data

    for detection in detections:
        # 박스 좌표와 신뢰도, 클래스 ID 가져오기
        xyxy = detection[:4].cpu().numpy().astype(int)
        conf = detection[4].item()
        cls = int(detection[5].item())
        
        # 클래스 라벨 설정
        label = category_names[cls]
        
        if conf > 0.30:  # 신뢰도가 50% 이상인 경우에만 표시
            # 박스 그리기
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            # 라벨 및 신뢰도 표시
            cv2.putText(frame, f'{label} {conf * 100:.2f}%', (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("Video Feed", frame)

    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

# 리소스 해제d

camera.release()
cv2.destroyAllWindows()


