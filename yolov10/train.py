from ultralytics import YOLO


model = YOLO()  # 기존 가중치 파일 경로


model.train(
    data='',  # 추가 데이터셋 설정 파일 경로
    epochs=300,  # 추가 학습 에포크 수 (필요에 따라 조정 가능)
    batch=32,   # 배치 크기 (원래 사용하던 크기 또는 데이터셋에 맞게 조정)
    name='solar_panel_detection',  # 새로운 실험 이름(모델이 저장될 디렉토리 이름)
    pretrained=False,  # 기존 가중치를 사용하여 학습 재개
    save_period=10  # 체크포인트 저장 주기 (에포크 수 기준)
)


results = model.val()

# 모델 저장
model.export(format='onnx')  # ONNX 형식으로 모델 내보내기
