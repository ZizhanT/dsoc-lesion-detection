import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob


def shuffle_and_split_dataset(data_dir, train_ratio=0.8):
    """
    打乱数据集并重新划分为训练集和验证集
    :param data_dir: 数据集根目录（包含 train 和 valid 文件夹）
    :param train_ratio: 训练集比例（默认 0.8）
    """
    # 获取所有图像和标签文件
    train_images = sorted(glob(os.path.join(data_dir, "train", "images", "*")))
    train_labels = sorted(glob(os.path.join(data_dir, "train", "labels", "*")))
    valid_images = sorted(glob(os.path.join(data_dir, "valid", "images", "*")))
    valid_labels = sorted(glob(os.path.join(data_dir, "valid", "labels", "*")))

    # 合并训练集和验证集
    all_images = train_images + valid_images
    all_labels = train_labels + valid_labels

    # 确保图像和标签文件一一对应
    assert len(all_images) == len(all_labels), "图像和标签文件数量不匹配"

    # 打乱数据集
    combined = list(zip(all_images, all_labels))
    train_data, val_data = train_test_split(combined, train_size=train_ratio, random_state=99)

    # 创建新的训练集和验证集文件夹
    new_train_dir = os.path.join(data_dir, "train_shuffled")
    new_val_dir = os.path.join(data_dir, "val_shuffled")
    os.makedirs(os.path.join(new_train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(new_train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, "labels"), exist_ok=True)

    # 保存训练集
    for img_path, label_path in train_data:
        # 复制图片文件
        shutil.copy(img_path, os.path.join(new_train_dir, "images", os.path.basename(img_path)))
        # 复制标签文件
        shutil.copy(label_path, os.path.join(new_train_dir, "labels", os.path.basename(label_path)))

    # 保存验证集
    for img_path, label_path in val_data:
        # 复制图片文件
        shutil.copy(img_path, os.path.join(new_val_dir, "images", os.path.basename(img_path)))
        # 复制标签文件
        shutil.copy(label_path, os.path.join(new_val_dir, "labels", os.path.basename(label_path)))

    print(f"数据集已重新划分并保存到 {new_train_dir} 和 {new_val_dir}")

if __name__ == "__main__":
    # 使用示例

    data_dir = "img"  # 数据集根目录
    shuffle_and_split_dataset(data_dir, train_ratio=0.9)