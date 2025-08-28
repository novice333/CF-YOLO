import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('..//runs/train/100 epoch VisDrone yolov11s_all4 result/weights/best.pt')
    model.predict(source='/DATA2T/1wlj/dataset/VisDrone/images/my_test2',
                  imgsz=640,
                  device='0',
                  save=True,
                  show_labels=False
                  )

