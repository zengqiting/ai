"""
胸部X光肺炎影像二分类 - 简单CNN版（无需下载）
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 数据集路径 ====================
data_dir = '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray'
print(f"数据集路径: {data_dir}")

# ==================== 数据预处理 ====================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]),
}

# ==================== 加载数据 ====================
full_train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=data_transforms['train']
)

# 按8:2划分
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = data_transforms['val']

test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'test'),
    transform=data_transforms['test']
)

print("=" * 50)
print("数据集统计:")
print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

train_labels = [full_train_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_train_dataset.targets[i] for i in val_dataset.indices]
test_labels = test_dataset.targets

print(f"训练集 - Normal: {train_labels.count(0)}, Pneumonia: {train_labels.count(1)}")
print(f"验证集 - Normal: {val_labels.count(0)}, Pneumonia: {val_labels.count(1)}")
print(f"测试集 - Normal: {test_labels.count(0)}, Pneumonia: {test_labels.count(1)}")
print("=" * 50)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 简单 CNN 模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = SimpleCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 训练配置 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ==================== 训练函数 ====================
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

# ==================== 训练 ====================
num_epochs = 15
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

# ==================== 测试 ====================
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

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# ==================== 绘图 ====================
os.makedirs('figures', exist_ok=True)

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

print("\n实验完成!")
print(f"最终测试准确率: {accuracy:.4f}")
