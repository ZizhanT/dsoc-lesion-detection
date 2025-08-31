from ultralytics import YOLO
from findLabelErrors import find_label_errors
from dataEnhance import get_augmentation_transform


def train():
    # 加载模型
    model = YOLO("best.pt")
    # 训练模型

    train_results = model.train(
        #data="yoloData.yaml", # 数据集 YAML 路径
        data="shuffledData.yaml",  # 数据集 YAML 路径
        epochs=2000,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device="cuda:0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
    )
    # python app_main.py

def main():
    train()

if __name__ == '__main__':
    main()
