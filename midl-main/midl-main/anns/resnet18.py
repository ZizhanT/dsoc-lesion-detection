import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # ✅ 适配无 GUI 服务器
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
batch_size = 16
num_epochs = 15
learning_rate = 0.001

# 数据集路径
data_dir_malignant = '/mnt/data/tqy/MedSAM-main/良恶性图片分类-3.5/恶性图片/images'  
data_dir_benign = '/mnt/data/tqy/MedSAM-main/良恶性图片分类-3.5/良性图片（疑似恶性终诊良性194张+恶性患者无病变图片）/images'

# 读取文件路径
malignant_files = [os.path.join(data_dir_malignant, f) for f in os.listdir(data_dir_malignant) if f.endswith('.jpg')]
benign_files = [os.path.join(data_dir_benign, f) for f in os.listdir(data_dir_benign) if f.endswith('.jpg')]
all_files = malignant_files + benign_files
labels = [1] * len(malignant_files) + [0] * len(benign_files)

# 9:1 划分训练集和验证集
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, labels, test_size=0.1, stratify=labels, random_state=42
)

# 数据集类
class CancerDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据增强和归一化
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = CancerDataset(train_files, train_labels, transform=train_transform)
val_dataset = CancerDataset(val_files, val_labels, transform=val_transform)

num_workers = min(4, os.cpu_count() - 1)  # ✅ 限制 workers 避免超载
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# 1️⃣ 确保不会自动下载
model = models.resnet18(weights=None)  # ✅ 这样才不会联网下载

# 2️⃣ 手动加载本地权重
weights_path = "/mnt/data/tqy/MedSAM-main/resnet18-f37072fd.pth"

# 确保权重文件存在
assert os.path.exists(weights_path), f"❌ 错误：找不到权重文件 {weights_path}"

# 读取权重
state_dict = torch.load(weights_path, map_location=device)

# 兼容不同格式
if "model" in state_dict:
    state_dict = state_dict["model"]

# 3️⃣ 加载权重
model.load_state_dict(state_dict, strict=False)


# 修改 ResNet18 最后一层（适应二分类任务）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练结果
train_loss_history = []
val_loss_history = []
val_auc_history = []
best_auc = 0.0

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).long()  # ✅ 确保是 long 类型

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    train_loss_history.append(epoch_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()  # ✅ 确保是 long 类型
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = val_correct / val_total
    val_auc = roc_auc_score(all_labels, all_probs)
    
    val_loss_history.append(val_epoch_loss)
    val_auc_history.append(val_auc)
    
    # 保存最佳模型
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, AUC: {val_auc:.4f}')
    print('-' * 50)

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(val_auc_history, label='Validation AUC')
plt.legend()
plt.title('AUC Curve')

plt.savefig("training_results.png")  # ✅ 适用于无 GUI 服务器

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 计算最终 AUC
probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        _, preds = torch.max(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print('Classification Report:')
print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
print('Confusion Matrix:')
print(confusion_matrix(all_labels, all_preds))
print(f'Final AUC: {roc_auc_score(all_labels, probs):.4f}')
