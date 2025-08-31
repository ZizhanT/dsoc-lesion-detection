import torch
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from ultralytics import YOLO


class FeatureHook:
    def __init__(self, model):
        self.model = model
        self.features = None
        # 注册钩子到backbone最后一层（需根据模型结构调整）
        self.hook = model.model.model[-2].register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output.detach()

    def get_feature(self, img_tensor):
        with torch.no_grad():
            self.model(img_tensor)  # 触发前向传播
        return self.features.mean(dim=[2, 3]).squeeze()  # 全局平均池化


class MyDataDistil:
    def __init__(self):
        self.name = 'apps.miod.my_data_distil.MyDataDistil'

    @staticmethod
    def extract_features(data_paths, model, device):
        """批量提取特征"""
        hook = FeatureHook(model)
        model.to(device).eval()
        features = []
        for path in tqdm(data_paths):
            img = cv2.imread(path)
            img = cv2.resize(img, (640, 640))  # 匹配模型输入尺寸
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255
            tensor = tensor.unsqueeze(0).to(device)
            feat = hook.get_feature(tensor)
            features.append(feat.cpu().numpy())
        return np.array(features)

    @staticmethod
    def find_swap_samples(train_feats, val_feats, n=100):
        """
        返回:
        - val_indices: 需要从验证集移到训练集的样本索引
        - train_indices: 需要从训练集移到验证集的样本索引
        """
        # 找验证集中最难样本（离训练集最远）
        nn_val = NearestNeighbors(n_neighbors=1).fit(train_feats)
        val_dists, _ = nn_val.kneighbors(val_feats)
        n = 50  # 取最远的n个样本
        hard_val_indices = np.argsort(val_dists.flatten())[-n:][::-1]  # 取距离最大的前n个

        # 找训练集中最冗余样本（离其他训练样本最近）
        nn_train = NearestNeighbors(n_neighbors=2).fit(train_feats)  # 计算到次近邻的距离
        train_dists, _ = nn_train.kneighbors(train_feats)
        redundancy_scores = train_dists[:, 1]  # 到最近非自身样本的距离
        base_pos = 100
        redundant_train_indices = np.argsort(redundancy_scores)[base_pos:base_pos + n]  # 取距离最近的n个
        return hard_val_indices, redundant_train_indices

    @staticmethod
    def perform_swap(train_paths, val_paths, swap_val_indices, swap_train_indices):
        """
        交换验证集和训练集中的数据
        :param train_paths: 训练集路径
        :param val_paths: 验证集路径
        :param swap_val_indices: 需要从验证集中移到训练集的索引
        :param swap_train_indices: 需要从训练集中移到验证集的索引
        :return: 更新后的训练集和验证集
        """
        new_train = [p for i, p in enumerate(train_paths) if i not in swap_train_indices]
        new_train += [val_paths[i] for i in swap_val_indices]

        new_val = [p for i, p in enumerate(val_paths) if i not in swap_val_indices]
        new_val += [train_paths[i] for i in swap_train_indices]

        return new_train, new_val


def main():
    # 假设已经训练好的模型在这里加载
    model = YOLO("best.pt")  # 训练好的模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练集和验证集路径
    train_paths = ["E:/medPicProject/img/train/images"]  # 训练集路径
    val_paths = ["E:/medPicProject/img/valid/images"]  # 验证集路径

    # 提取特征
    train_feats = MyDataDistil.extract_features(train_paths, model, device)
    val_feats = MyDataDistil.extract_features(val_paths, model, device)

    # 查找最难和最冗余样本
    hard_val_indices, redundant_train_indices = MyDataDistil.find_swap_samples(train_feats, val_feats, n=50)

    # 交换数据
    new_train, new_val = MyDataDistil.perform_swap(train_paths, val_paths, hard_val_indices, redundant_train_indices)

    # 输出新数据集路径
    print(f"Updated train dataset: {new_train}")
    print(f"Updated val dataset: {new_val}")

    # 生成新的 YAML 文件，更新路径为交换后的路径
    with open("shuffledData.yaml", 'w') as f:
        f.write(f"path: E:/medPicProject/img\n")
        f.write(f"train: {new_train}\n")
        f.write(f"val: {new_val}\n")
        f.write(f"test: {new_val}\n")  # test set同样可以使用更新后的val路径
        f.write(f"nc: 3\n")
        f.write("names:\n")
        f.write("  0: irregular_nodulation\n")
        f.write("  1: irregular_surface\n")
        f.write("  2: irregular_vascularit\n")

    # 使用更新后的 YAML 进行训练
    model.train(data="shuffledData.yaml", epochs=2000, imgsz=640, device="cuda:0")


if __name__ == '__main__':
    main()
