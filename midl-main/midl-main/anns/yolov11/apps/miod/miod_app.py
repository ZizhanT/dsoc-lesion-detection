#
from typing import Dict
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
from ultralytics import solutions

class MiodApp(object):
    def __init__(self):
        self.name = 'apps.miod_app.MiodApp'

    @staticmethod
    def startup(params:Dict = {}) -> None:
        print(f'医学图像目标检测 v0.0.1')
        # MiodApp.train_main(params=params)
        # MiodApp.predict_main(params=params)
        # MiodApp.evaluate_main(params=params)
        MiodApp.gen_heatmap(params=params)


    @staticmethod
    def train_main(params:Dict = {}) -> None:
        # 加载模型
        model = YOLO("./work/ckpts/yolo11n.pt")
        # 训练模型
        train_results = model.train(
            data="./datasets/xjd_med/t001.yaml",  # 数据集 YAML 路径
            epochs=500,  # 训练轮次
            imgsz=640,  # 训练图像尺寸
            device="cuda:2",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
        )

    @staticmethod
    def evaluate_main(params:Dict = {}) -> None:
        # 加载模型
        model = YOLO("runs/detect/train7/weights/best.pt")
        # 评估模型在验证集上的性能
        metrics = model.val() # metrics: ultralytics.utils.metrics.DetMetrics

    @staticmethod
    def predict_main(params:Dict = {}) -> None:
        # 加载模型
        model = YOLO("runs/detect/train4/weights/best.pt")
        test_img_rfn = 'datasets/xjd_med/train/images/-_mp4-0209_jpg.rf.aef529f07213a83adf2be894393ddb98.jpg'
        label_rfn = 'datasets/xjd_med/train/labels/-_mp4-0209_jpg.rf.aef529f07213a83adf2be894393ddb98.txt'
        results = model(test_img_rfn) # results: List(len=1) [0] = ultralytics.engine.results.Results
        print(f'### results: {len(results)}; {type(results[0])};')
        # 绘制网络输出的检测框
        results[0].plot(show=False, save=True, filename='./work/a001.jpg')
        # 显示网络输出的检测框内容
        net_boxes = results[0].boxes.data[:, :4]
        for r in results:
            print(r.boxes.data)  # print detection bounding boxes
        # 画Ground Truth检测框
        img_obj = cv2.imread(test_img_rfn)
        img_h, img_w, chn = img_obj.shape
        color = (255, 0, 0)
        thickness = 2
        gt_boxes = None
        with open(label_rfn, 'r', encoding='utf-8') as rfd:
            for row in rfd:
                row = row.strip()
                print(f'### {row};')
                arrs0 = row.split(' ')
                cls_id = arrs0[0]
                if gt_boxes is None:
                    gt_boxes = torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])
                else:
                    gt_boxes = torch.vstack((gt_boxes, torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])))
                cx, cy, bw, bh = float(arrs0[1])*img_w, float(arrs0[2])*img_h, float(arrs0[3])*img_w, float(arrs0[4])*img_h
                top_left_x = int(cx - bw/2.0)
                top_left_y = int(cy - bh/2.0)
                bottom_right_x = int(cx + bw/2.0)
                bottom_right_y = int(cy + bh/2.0)
                cv2.rectangle(img_obj, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
        cv2.imwrite('./work/a002.jpg', img_obj)
        gt_boxes = gt_boxes.to(net_boxes.device)
        # 计算IOU
        print(f'Net: {net_boxes.shape}; GT: {gt_boxes.shape};')
        ious = bbox_iou(net_boxes, gt_boxes)
        print(f'ious: {ious.shape}; \n{ious};')

    @staticmethod
    def gen_heatmap(params:Dict = {}) -> None:
        test_img_rfn = 'datasets/xjd_med/train/images/-_mp4-0209_jpg.rf.aef529f07213a83adf2be894393ddb98.jpg'
        label_rfn = 'datasets/xjd_med/train/labels/-_mp4-0209_jpg.rf.aef529f07213a83adf2be894393ddb98.txt'
        img_obj = cv2.imread(test_img_rfn)
        # Initialize heatmap object
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,  # Color of the heatmap
            show=True,  # Display the image during processing
            model="runs/detect/train7/weights/best.pt",  # Ultralytics YOLO11 model file
        )
        # Generate heatmap on the frame
        ht_img = heatmap_obj.generate_heatmap(img_obj)
        img_h, img_w, chn = ht_img.shape
        color = (255, 0, 0)
        thickness = 2
        gt_boxes = None
        with open(label_rfn, 'r', encoding='utf-8') as rfd:
            for row in rfd:
                row = row.strip()
                print(f'### {row};')
                arrs0 = row.split(' ')
                cls_id = arrs0[0]
                if gt_boxes is None:
                    gt_boxes = torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])
                else:
                    gt_boxes = torch.vstack((gt_boxes, torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])))
                cx, cy, bw, bh = float(arrs0[1])*img_w, float(arrs0[2])*img_h, float(arrs0[3])*img_w, float(arrs0[4])*img_h
                top_left_x = int(cx - bw/2.0)
                top_left_y = int(cy - bh/2.0)
                bottom_right_x = int(cx + bw/2.0)
                bottom_right_y = int(cy + bh/2.0)
                cv2.rectangle(ht_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
        cv2.imwrite('./work/a003.jpg', ht_img)





    
