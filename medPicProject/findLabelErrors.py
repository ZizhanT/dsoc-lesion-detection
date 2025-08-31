from ultralytics import YOLO
import os
import cv2
from pathlib import Path
import csv
from datetime import datetime



def find_label_errors(model_path, val_images_dir, val_labels_dir, conf_thres=0.5, iou_thres=0.5):
    """
    检测验证集中标签错误的图片
    参数：
        model_path: 训练好的模型路径
        val_images_dir: 验证集图片目录
        val_labels_dir: 验证集标签目录
        conf_thres: 置信度阈值
        iou_thres: IOU匹配阈值
    返回：
        error_records: 错误记录列表，包含文件路径、预测结果和真实标签
    """
    # 加载模型
    model = YOLO("yolo11n.pt")
    error_records = []

    # 遍历验证集图片
    for img_path in Path(val_images_dir).glob('*.*'):
        # 预测结果
        results = model.predict(img_path, conf=conf_thres, iou=iou_thres)

        # 读取真实标签
        label_path = Path(val_labels_dir) / f"{img_path.stem}.txt"
        true_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                true_labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        # 提取预测结果
        pred_labels = []
        for box in results[0].boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            xywhn = box.xywhn.cpu().numpy()[0]  # 归一化坐标
            pred_labels.append([cls, conf, *xywhn])

        # 标签比对逻辑
        errors = []
        for true_label in true_labels:
            true_cls = int(true_label[0])
            matched = False

            # 简单匹配逻辑（可根据需要扩展IOU计算）
            for pred_label in pred_labels:
                pred_cls = pred_label[0]
                # 如果预测类别匹配且IOU>阈值（此处简化为坐标相近判断）
                if pred_cls == true_cls and abs(pred_label[2] - true_label[1]) < 0.1:
                    matched = True
                    break

            if not matched:
                errors.append({
                    'true_class': true_cls,
                    'pred_classes': [p[0] for p in pred_labels],
                    'coordinates': true_label[1:]
                })

        # 记录错误
        if errors:
            error_records.append({
                'image_path': str(img_path),
                'errors': errors,
                'pred_results': pred_labels,
                'true_labels': true_labels
            })

    return error_records

def save_error_report(error_records, output_file="error_report.txt", file_format="txt"):
    """
    保存错误报告为文件（支持txt或csv格式）
    参数：
        error_records: 错误记录列表（来自find_label_errors函数）
        output_file: 输出文件名
        file_format: 文件格式，支持 "csv" 或 "txt"
    返回：
        output_file: 保存的文件路径
    """
    if file_format not in ["csv", "txt"]:
        raise ValueError("file_format 必须是 'csv' 或 'txt'")

    # 打开文件并写入数据
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        if file_format == "csv":
            writer = csv.writer(file)
            # 写入表头
            writer.writerow([
                "图片名称", "错误类型", "真实类别", "预测类别", "标注坐标", "预测坐标"
            ])
            # 写入数据
            for record in error_records:
                image_name = Path(record['image_path']).name
                for error in record['errors']:
                    # 检查 pred_results 是否有效
                    pred_coords = []
                    if 'pred_results' in record and record['pred_results']:
                        for p in record['pred_results']:
                            if len(p) >= 4:  # 确保有足够的维度
                                pred_coords.append(p[2:])  # 提取坐标部分
                    writer.writerow([
                        image_name,
                        "漏检" if error['true_class'] not in error['pred_classes'] else "误检",
                        error['true_class'],
                        error['pred_classes'],
                        error['coordinates'],
                        pred_coords  # 预测坐标
                    ])
        else:  # txt格式
            file.write(f"错误报告生成时间: {datetime.now()}\n\n")
            for i, record in enumerate(error_records, 1):
                image_name = Path(record['image_path']).name
                file.write(f"{i}. 图片名称: {image_name}\n")
                for j, error in enumerate(record['errors'], 1):
                    file.write(f"   错误 {j}:\n")
                    file.write(f"     - 错误类型: {'漏检' if error['true_class'] not in error['pred_classes'] else '误检'}\n")
                    file.write(f"     - 真实类别: {error['true_class']}\n")
                    file.write(f"     - 预测类别: {error['pred_classes']}\n")
                    file.write(f"     - 标注坐标: {error['coordinates']}\n")
                    # 检查 pred_results 是否有效
                    if 'pred_results' in record and record['pred_results']:
                        pred_coords = []
                        for p in record['pred_results']:
                            if len(p) >= 4:  # 确保有足够的维度
                                pred_coords.append(p[2:])  # 提取坐标部分
                        file.write(f"     - 预测坐标: {pred_coords}\n")
                    else:
                        file.write("     - 预测坐标: 无有效预测结果\n")
                    file.write("\n")

    print(f"错误报告已保存至: {output_file}")
    return output_file

if __name__ == "__main__":
    # errors = find_label_errors(
    #     model_path="yolo11n.pt",  # 你的训练模型
    #     val_images_dir="img/valid/images",
    #     val_labels_dir="img/valid/labels",
    #     conf_thres=0.5
    # )

    # 打印错误报告
    # 假设已经通过 find_label_errors 函数获取了错误记录
    errors = find_label_errors(
        model_path="yolo11n.pt",
        val_images_dir="img/valid/images",
        val_labels_dir="img/valid/labels",
        conf_thres=0.5
    )

    # 保存为CSV文件
    save_error_report(errors, output_file="error_report.csv", file_format="csv")

    # 保存为TXT文件
    save_error_report(errors, output_file="error_report.txt", file_format="txt")
