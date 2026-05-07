"""
胸部X光肺炎影像二分类
数据集：Chest X-Ray Images (Pneumonia)
任务：Normal vs Pneumonia 二分类
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 1. 数据预处理 ====================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
}

# ==================== 2. 加载数据 ====================
# 注意：Kaggle上路径为 '/kaggle/input/chest-xray-pneumonia/chest_xray'
# 本地请修改为你的数据集路径
data_dir = 'chest_xray'  # 本地路径
if os.path.exists('/kaggle/input/chest-xray-pneumonia/chest_xray'):
    data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'

# 加载完整训练集
full_train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=data_transforms['train']
)

# 按8:2划分训练集和验证集
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 为验证集单独设置transform
val_dataset.dataset.transform = data_transforms['val']

# 加载测试集
test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'test'),
    transform=data_transforms['test']
)

print("=" * 50)
print("数据集统计:")
print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")
print("=" * 50)

# 统计各类别数量
train_labels = [full_train_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_train_dataset.targets[i] for i in val_dataset.indices]
test_labels = test_dataset.targets

print(f"训练集 - Normal: {train_labels.count(0)}, Pneumonia: {train_labels.count(1)}")
print(f"验证集 - Normal: {val_labels.count(0)}, Pneumonia: {val_labels.count(1)}")
print(f"测试集 - Normal: {test_labels.count(0)}, Pneumonia: {test_labels.count(1)}")
print("=" * 50)

# 创建DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ==================== 3. 迁移学习模型 ====================
def create_resnet_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 冻结底层
    for param in model.parameters():
        param.requires_grad = False
    # 替换全连接层
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

model = create_resnet_model().to(device)

print("模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 4. 训练配置 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ==================== 5. 训练函数 ====================
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# ==================== 6. 训练循环 ====================
num_epochs = 20
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n" + "=" * 50)
print("开始训练")
print("=" * 50)

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = validate()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f'\nEpoch {epoch}/{num_epochs}:')
    print(f'训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    print("-" * 50)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'保存最佳模型 (验证准确率: {val_acc:.2f}%)')

total_time = time.time() - start_time
print(f"\n训练完成! 总耗时: {total_time:.2f}秒")

# ==================== 7. 测试集评估 ====================
print("\n" + "=" * 50)
print("测试集评估")
print("=" * 50)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# ==================== 8. 绘制图表 ====================
# 确保figures目录存在
os.makedirs('figures', exist_ok=True)

# 8.1 训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练与验证损失曲线')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('训练与验证准确率曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('figures/training_curves.png', dpi=150)
plt.show()

# 8.2 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('测试集混淆矩阵')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("实验完成!")
print(f"图表已保存到 figures/ 文件夹")
print("=" * 50)
