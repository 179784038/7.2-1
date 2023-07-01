import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import transforms

# 设置随机种子
torch.manual_seed(42)

# 定义超参数
batch_size = 64
learning_rate = 0.1
num_epochs = 100

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载CIFAR-10数据集并进行预处理
transform_train = transforms.Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    transforms.ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform_test)

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义ResNet-18模型
model = resnet18(num_classes=10)

# 将模型移动到设备
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 记录训练过程的指标
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# 使用TensorBoard可视化训练过程
writer = SummaryWriter(log_dir='./logs')

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        # 将输入和标签数据加载到设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练损失
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每10个批次记录一次训练损失
        if (i + 1) % 10 == 0:
            avg_train_loss = train_loss / 10
            avg_train_acc = 100.0 * correct / total

            train_loss_history.append(avg_train_loss)
            train_acc_history.append(avg_train_acc)

            writer.add_scalar('Train Loss', avg_train_loss, epoch * total_step + i + 1)
            writer.add_scalar('Train Accuracy', avg_train_acc, epoch * total_step + i + 1)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.2f}%')

            train_loss = 0.0
            correct = 0
            total = 0

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100.0 * val_correct / val_total

    val_loss_history.append(avg_val_loss)
    val_acc_history.append(avg_val_acc)

    writer.add_scalar('Val Loss', avg_val_loss, epoch + 1)
    writer.add_scalar('Val Accuracy', avg_val_acc, epoch + 1)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%')

    scheduler.step()

# 在测试集上评估模型
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_accuracy = 100.0 * test_correct / test_total

print('Test Accuracy: {:.2f}%'.format(test_accuracy))

# 保存模型
torch.save(model.state_dict(), 'resnet18_cifar10.pth')

# 关闭TensorBoard写入器
writer.close()

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
