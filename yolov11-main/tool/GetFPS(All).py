import torch
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from PIL import Image
import time

# 模型路径和测试集路径
model_path = '/DATA2T/1wlj/0TransForms/yolov11-main/test/YOLOV11/best.pt'
test_images_dir = Path('/DATA2T/1wlj/dataset/TinyPerson/images/test')

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

# 获取所有测试图像路径
image_paths = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))

def calculate_fps(model, image_paths):
    total_time = 0
    num_images = len(image_paths)

    for img_path in image_paths:
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(img).unsqueeze(0)  # 添加批次维度

        # 开始计时
        start_time = time.time()

        # 进行推理
        with torch.no_grad():
            _ = model(img_tensor)

        # 结束计时
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

    # 计算平均FPS
    average_fps = num_images / total_time if total_time > 0 else 0
    return average_fps

fps = calculate_fps(model, image_paths)
print(f'Average FPS: {fps:.2f}')