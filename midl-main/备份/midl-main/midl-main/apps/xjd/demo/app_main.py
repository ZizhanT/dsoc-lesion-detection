# app.py
import sys
sys.path.append('./anns/yolov11')
from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import os
import base64
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

model = None
resnet18_model = None
device = 'cuda:0'
total_frames, finished_frames = 0, 0
fig1_url = None
fig1_benign_x, fig1_benign_y = [], []
fig1_malignant_x, fig1_malignant_y = [], []
frame_seq = 0

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def detect(frame):
    '''
    在视频帧上绘制检测框
    '''
    results = model(frame) # results: List(len=1) [0] = ultralytics.engine.results.Results
    for i, d in enumerate(reversed(results[0].boxes)):
        cv2.rectangle(frame, 
            (int(d.xyxy[0][0]), int(d.xyxy[0][1])), 
            (int(d.xyxy[0][2]), int(d.xyxy[0][3])), 
            (0, 255, 0), 2
        )
        if d.cls == 0:
            cls_name = 'c1'
        elif d.cls == 1:
            cls_name = 'c2'
        elif d.cls == 2:
            cls_name = 'c3'
        print(f'################ conf: {type(d.conf)}; {d.conf};')
        cv2.putText(
            frame, f"{cls_name}: {d.conf.detach().cpu().item():0.2f}", (int(d.xyxy[0][0]), int(d.xyxy[0][1])), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )


def classify(frame):
    global frame_seq
    global fig1_x_raw, fig1_y_raw
    global fig1_benign_x, fig1_benign_y
    global fig1_malignant_x, fig1_malignant_y
    X = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
    X = X.to(device)
    print(f'X: {X.shape};')
    results = resnet18_model(X)
    print(f'classify 1')
    max_val = torch.max(results)
    print(f'!!! max_val: {type(max_val)}; {max_val};')
    results = results - max_val
    print(f'!!! processed results: {results};')
    exps = torch.exp(results)
    print(f'classify 2 exp[0][0]: {exps[0][0]} , {exps[0][1]}; {type(exps[0][0])};')
    prob = exps[0][0].detach().cpu().item() / (exps[0][0].detach().cpu().item() + exps[0][1].detach().cpu().item())
    print(f'classify 3 prob: {prob}; {type(prob)};')
    frame_seq += 1
    if results[0][0] > results[0][1]:
        fig1_benign_x.append(frame_seq)
        fig1_benign_y.append(prob)
        cv2.putText(
            frame, f"Benign: {prob:0.2f}", (0, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )
    else:
        fig1_malignant_x.append(frame_seq)
        fig1_malignant_y.append(1-prob)
        cv2.putText(
            frame, f"Malignant: {1-prob:0.2f}", (0, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )
    print(f'!!!!!!!! classify results: {results.shape}; {results};')

def process_frame(frame):
    """示例处理函数：在帧上绘制检测框并添加置信度"""
    height, width = frame.shape[:2]
    detect(frame)
    classify(frame)
    # cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
    # cv2.putText(frame, "Detection: 95%", (50, 40), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getProgress', methods=['POST'])
def get_progress():
    '''
    更新进度框中的内容
    '''
    global total_frames, finished_frames
    global fig1_url
    global fig1_benign_x, fig1_benign_y
    global fig1_malignant_x, fig1_malignant_y
    if total_frames <= 0:
        return jsonify({
            'progress': f'视频上传处理中，请稍候...',
            'state': 'uploading'
        })
    elif finished_frames >= total_frames:
        # 绘制散点图
        plt.scatter(np.array(fig1_benign_x), np.array(fig1_benign_y))
        plt.scatter(np.array(fig1_malignant_x), np.array(fig1_malignant_y))
        plt.savefig('./static/fig1.jpg')
        fig1_url = 'static/fig1.jpg'
        return jsonify({
            'progress': f'处理完成，初始化可视化界面，请稍候...',
            'state': 'finished'
        })
    else:
        return jsonify({
            'progress': f'处理进度：{finished_frames} / {total_frames}',
            'state': 'processing'
        })
    
@app.route('/getFig1Url', methods=['POST'])
def get_fig1_url():
    '''
    获取统计图表1的URL
    '''
    global fig1_url
    if fig1_url is None:
        return jsonify({
            'fig1_url': ''
        })
    else:
        return jsonify({
            'fig1_url': fig1_url
        })

@app.route('/upload', methods=['POST'])
def upload_video():
    global model, resnet18_model
    global total_frames, finished_frames
    global device
    # 载入yolov11模型
    model = YOLO("anns/yolov11/best.pt")
    model = model.to(device)
    resnet18_model = models.resnet18(weights=None)
    num_ftrs = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(num_ftrs, 2)
    weights_path = 'anns/resnet_best_model.pth'
    state_dict = torch.load(weights_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    resnet18_model.load_state_dict(state_dict, strict=False)
    resnet18_model = resnet18_model.to(device)
    resnet18_model.eval()
    # 
    if 'video' not in request.files:
        return jsonify({'error': 'No video selected'}), 400
    
    video = request.files['video']
    temp_dir = tempfile.mkdtemp()
    upload_path = os.path.join(temp_dir, video.filename)
    video.save(upload_path)

    cap = cv2.VideoCapture(upload_path)
    frames = []
    processed_frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed)
        frames.append(base64.b64encode(buffer).decode('utf-8'))
        processed_frames.append(processed)
        finished_frames += 1

    cap.release()
    
    # 生成结果视频
    result_filename = f'processed_{video.filename}'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out.write(frame)
    out.release()

    shutil.rmtree(temp_dir)

    return jsonify({
        'frames': frames,
        'diagnosis': '良性（95%）',
        'video_url': f'/download/{result_filename}'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)