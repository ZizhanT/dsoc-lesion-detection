import os
import cv2
import numpy as np
import base64
import shutil
import tempfile
import torch
import torch.nn as nn
import threading
from flask import Flask, request, jsonify, send_from_directory, render_template
from torchvision import models
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs('static/figs', exist_ok=True)

device = 'cuda:0'
model = None
resnet18_model = None

# 全局统计数据
total_frames, finished_frames = 0, 0
frame_seq = 0
benign_probs, malignant_probs = [], []
benign_x, malignant_x = [], []
confidence_list = []
benign_count, malignant_count = 0, 0

# 页面路由
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/team')
def team():
    return render_template('team.html')

# 后台处理视频的逻辑（新开的线程来跑这个！）
def background_process_video(video_path, result_filename):
    global model, resnet18_model
    global total_frames, finished_frames, frame_seq
    global benign_probs, malignant_probs, benign_x, malignant_x, confidence_list
    global benign_count, malignant_count

    try:
        finished_frames = 0
        frame_seq = 0
        benign_probs, malignant_probs = [], []
        benign_x, malignant_x = [], []
        confidence_list = []
        benign_count, malignant_count = 0, 0

        # 加载模型（确保每次都重新加载，避免线程冲突）
        model = YOLO(os.path.join(BASE_DIR, "anns/yolov11/yolov11_best.pt")).to(device)

        resnet18_model = models.resnet18(weights=None)
        resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 2)
        state_dict = torch.load(os.path.join(BASE_DIR, 'anns/resnet_best_model.pth'), map_location=device)
        resnet18_model.load_state_dict(state_dict.get("model", state_dict), strict=False)
        resnet18_model = resnet18_model.to(device).eval()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0 or width == 0 or height == 0:
            print("[Error] 视频读取失败")
            return

        processed_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            processed_frames.append(processed)
            finished_frames += 1
        cap.release()

        # 保存处理后视频
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in processed_frames:
            out.write(f)
        out.release()

        generate_all_figs()

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 处理完后把上传的临时视频删掉
        if os.path.exists(video_path):
            os.remove(video_path)

# 上传接口（只保存视频，不处理！）
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': '没有上传视频文件'}), 400

        video = request.files['video']
        temp_dir = tempfile.mkdtemp()
        upload_path = os.path.join(temp_dir, video.filename)
        video.save(upload_path)

        # 准备处理后保存的文件名
        result_filename = f'processed_{video.filename}'

        # 开后台线程去处理
        t = threading.Thread(target=background_process_video, args=(upload_path, result_filename))
        t.start()

        return jsonify({'message': '视频上传成功，开始处理！'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'上传处理出错：{str(e)}'}), 500

# 帧处理函数
def process_frame(frame):
    global frame_seq, benign_probs, malignant_probs, confidence_list
    global benign_x, malignant_x, benign_count, malignant_count

    frame_seq += 1
    detect(frame)

    X = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        results = resnet18_model(X)[0]
    max_val = torch.max(results)
    results -= max_val
    probs = torch.exp(results)
    prob = probs[0].item() / (probs[0].item() + probs[1].item())
    confidence_list.append(prob)

    if results[0] > results[1]:
        benign_probs.append(prob)
        benign_x.append(frame_seq)
        benign_count += 1
        label = f"Benign: {prob:.2f}"
    else:
        malignant_probs.append(1 - prob)
        malignant_x.append(frame_seq)
        malignant_count += 1
        label = f"Malignant: {1 - prob:.2f}"

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# 检测器
def detect(frame):
    results = model(frame)
    for d in results[0].boxes:
        cls_name = ['c1', 'c2', 'c3'][int(d.cls)]
        cv2.rectangle(frame, (int(d.xyxy[0][0]), int(d.xyxy[0][1])), (int(d.xyxy[0][2]), int(d.xyxy[0][3])), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name}: {d.conf.item():.2f}", (int(d.xyxy[0][0]), int(d.xyxy[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

# 生成全部图表
def generate_all_figs():
    x_range = list(range(1, frame_seq + 1))
    os.makedirs('static/figs', exist_ok=True)

    plt.figure()
    plt.scatter(benign_x, benign_probs, c='green')
    plt.title('Benign Probability vs Frame')
    plt.xlabel('Frame')
    plt.ylabel('Benign Probability')
    plt.savefig('static/figs/fig1.jpg')
    plt.close()

    plt.figure()
    plt.scatter(malignant_x, malignant_probs, c='red')
    plt.title('Malignant Probability vs Frame')
    plt.xlabel('Frame')
    plt.ylabel('Malignant Probability')
    plt.savefig('static/figs/fig2.jpg')
    plt.close()

    plt.figure()
    plt.bar(['Benign', 'Malignant'], [benign_count, malignant_count], color=['green', 'red'])
    plt.title('Classification Count')
    plt.savefig('static/figs/fig3.jpg')
    plt.close()

    plt.figure()
    plt.hist(confidence_list, bins=10, color='blue')
    plt.title('Classification Confidence Histogram')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.savefig('static/figs/fig4.jpg')
    plt.close()

    if len(benign_x) >= 5:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(benign_probs, size=5)
        plt.figure()
        plt.plot(benign_x, smoothed, color='green')
        plt.title('Smoothed Benign Probability')
        plt.savefig('static/figs/fig5.jpg')
        plt.close()

    plt.figure()
    benign_cum = np.cumsum([1 if i in benign_x else 0 for i in x_range])
    malignant_cum = np.cumsum([1 if i in malignant_x else 0 for i in x_range])
    plt.plot(x_range, benign_cum, label='Benign', color='green')
    plt.plot(x_range, malignant_cum, label='Malignant', color='red')
    plt.legend()
    plt.title('Cumulative Prediction Count')
    plt.savefig('static/figs/fig6.jpg')
    plt.close()

# 返回图表
@app.route('/getAllFigs', methods=['POST'])
def get_all_figs():
    fig_urls = [f'/static/figs/fig{i}.jpg' for i in range(1, 7)]
    return jsonify({'fig_urls': fig_urls})

# 返回处理进度
@app.route('/getProgress', methods=['POST'])
def get_progress():
    global total_frames, finished_frames
    if total_frames <= 0:
        return jsonify({'progress': '视频上传处理中，请稍候...', 'state': 'uploading'})
    elif finished_frames >= total_frames:
        return jsonify({'progress': '处理完成，初始化可视化界面，请稍候...', 'state': 'finished', 'ratio': 100})
    else:
        ratio = int(finished_frames / total_frames * 100)
        return jsonify({
            'progress': f'处理进度：{finished_frames} / {total_frames}',
            'state': 'processing',
            'ratio': ratio
        })

# 下载视频
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
