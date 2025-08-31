# 试验程序
import sys
sys.path.append('./anns/yolov11')
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

model = None
resnet18_model = None

def upload_video():
    global model, resnet18_model
    print(f'视频上传')# 加载模型
    model = YOLO("anns/yolov11/weights/yolov11_best.pt")
    resnet18_model = models.resnet18(weights=None)
    weights_path = 'anns/resnet_best_model.pth'
    device = 'cpu'
    state_dict = torch.load(weights_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    resnet18_model.load_state_dict(state_dict, strict=False)
    num_ftrs = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(num_ftrs, 2)
    resnet18_model = resnet18_model.to(device)
    resnet18_model.eval()
    # 
    cap = cv2.VideoCapture('videos/video_001.mp4')
    frames = []
    processed_frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f'frame: {type(frame)};')
        process_frame(frame)
        break
    cap.release()

def process_frame(frame):
    detect(frame)
    classify(frame)

def classify(frame):
    results = resnet18_model(frame)
    print(f'!!!!!!!! classify results: {results.shape}; {results};')

def detect(frame):
    results = model(frame) # results: List(len=1) [0] = ultralytics.engine.results.Results
    print(f'### results: {len(results)}; {type(results[0])};')
    # 绘制网络输出的检测框
    # results[0].plot(show=False, save=True, filename='./work/a001.jpg')
    # print(f'{results[0]};')
    # 显示网络输出的检测框内容
    for i, d in enumerate(reversed(results[0].boxes)):
        print(f'### {d.cls}:{d.conf}: {d.xyxy}; frame: {frame.shape};')
        cv2.rectangle(frame, 
            (0, 0), 
            (frame.shape[1], frame.shape[0]), 
            (0, 0, 255), 2
        )
        cv2.rectangle(frame, 
            (int(d.xyxy[0][0]), int(d.xyxy[0][1])), 
            (int(d.xyxy[0][2]), int(d.xyxy[0][3])), 
            (0, 255, 0), 2
        )
    cv2.imwrite('./work/b001.jpg', frame)
    # net_boxes = results[0].boxes.data[:, :4]
    # for r in results:
    #     print(r.boxes.data)  # print detection bounding boxes
    # # 画Ground Truth检测框
    # img_obj = cv2.imread(test_img_rfn)
    # img_h, img_w, chn = img_obj.shape
    # color = (255, 0, 0)
    # thickness = 2
    # gt_boxes = None
    # with open(label_rfn, 'r', encoding='utf-8') as rfd:
    #     for row in rfd:
    #         row = row.strip()
    #         print(f'### {row};')
    #         arrs0 = row.split(' ')
    #         cls_id = arrs0[0]
    #         if gt_boxes is None:
    #             gt_boxes = torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])
    #         else:
    #             gt_boxes = torch.vstack((gt_boxes, torch.tensor([[float(arrs0[1]), float(arrs0[2]), float(arrs0[3]), float(arrs0[4])]])))
    #         cx, cy, bw, bh = float(arrs0[1])*img_w, float(arrs0[2])*img_h, float(arrs0[3])*img_w, float(arrs0[4])*img_h
    #         top_left_x = int(cx - bw/2.0)
    #         top_left_y = int(cy - bh/2.0)
    #         bottom_right_x = int(cx + bw/2.0)
    #         bottom_right_y = int(cy + bh/2.0)
    #         cv2.rectangle(img_obj, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
    # cv2.imwrite('./work/a002.jpg', img_obj)
    # gt_boxes = gt_boxes.to(net_boxes.device)

def main():
    upload_video()
    print(f'^_^ The End! ^_^')

if '__main__' == __name__:
    main()