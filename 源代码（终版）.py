# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os,shutil
import random
import time

# %%
import os
folder_path = r"F:\实验室\猫狗分类\dc\train"
print("路径是否存在:", os.path.exists(folder_path))
print("路径下的文件:", os.listdir(folder_path))

# %%
import os, shutil 
# 原始目录所在的路径
original_dataset_dir = r"F:\实验室\猫狗分类\dc\train"

# 数据集分类后的目录
base_dir = r"F:\实验室\猫狗分类\dc\train-cats-and-dogs"
os.makedirs(base_dir, exist_ok=True)

# # 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# 猫训练图片所在目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_cats_dir, exist_ok=True)

# 狗训练图片所在目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_dogs_dir, exist_ok=True)

# 猫验证图片所在目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.makedirs(validation_cats_dir, exist_ok=True)

# 狗验证数据集所在目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(validation_dogs_dir, exist_ok=True)

# 猫测试数据集所在目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.makedirs(test_cats_dir, exist_ok=True)

# 狗测试数据集所在目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.makedirs(test_dogs_dir, exist_ok=True)

# 将前1000张猫图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")

# 将下500张猫图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")
    
# 将下500张猫图像复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")
    
# 将前1000张狗图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")
    
# 将下500张狗图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")
    
# 将下500张狗图像复制到test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"File not found: {src}")

# %%
# 数据路径设置
train_datadir = r"F:\实验室\猫狗分类\dc\train-cats-and-dogs\train"  # 包含cat和dog子目录
test_datadir = r"F:\实验室\猫狗分类\dc\train-cats-and-dogs\test"    # 包含cat和dog子目录

# %%
# 数据增强和预处理
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_data = datasets.ImageFolder(train_datadir, transform=train_transforms)
test_data = datasets.ImageFolder(test_datadir, transform=test_transforms)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)


# %%
# 可视化数据增强效果
def im_convert(tensor):
    """反标准化并转换为可显示格式"""
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    return image

# 可视化样本
def visualize_samples(loader, classes, num_samples=8):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(15, 5))
    for idx in range(num_samples):
        ax = fig.add_subplot(2, num_samples//2, idx+1, xticks=[], yticks=[])
        ax.imshow(im_convert(images[idx]))
        ax.set_title(classes[labels[idx].item()])
    plt.show()

visualize_samples(train_loader, train_data.classes)

# %%
# 使用ResNet18迁移学习
class CatDogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        # 冻结卷积层参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 修改最后的全连接层
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.base_model(x)

# %%
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogClassifier().to(device)
print(model)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# 训练函数
def train_model(epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-"*60)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_data)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

         # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_running_loss / len(test_data)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}\n")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()
    
    return history

# 开始训练
history = train_model(epochs=15)

# %%
# 模型评估
def evaluate_model():
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 生成分类报告
    print(classification_report(all_labels, all_preds, target_names=test_data.classes))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_data.classes, 
                yticklabels=test_data.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model()

# %%
# 单样本预测演示
def predict_image(image_path):
    from PIL import Image
    
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title(f"Predicted: {test_data.classes[predicted.item()]} ({probability[predicted.item()]:.1f}%)")
    plt.show()

# 示例预测（需要替换为实际图片路径）
predict_image(r"F:\实验室\猫狗分类\dc\train-cats-and-dogs\validation\cats\cat.1000.jpg" )


