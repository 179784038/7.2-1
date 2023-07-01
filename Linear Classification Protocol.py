import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import math
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('.', train=True, download=download, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10('.', train=False, download=download, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)

    return train_loader, test_loader

model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)

checkpoint = torch.load('input/self-supervised/runs/Jun26_08-59-39_87b8ae8429c8/checkpoint_0050.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
    if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

cifar10_train_loader, cifar10_test_loader = get_cifar10_data_loaders(download=True)

# Freeze all layers except the last
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 50
top1_train_accuracy_list = [0]
top1_accuracy_list = [0]
top5_accuracy_list = [0]
epoch_list = [0]

writer = SummaryWriter(log_dir='logs')  # TensorBoard writer

train_loss_list = []

for epoch in range(epochs):
    top1_train_accuracy = 0
    total_loss = 0  # Track total loss during training

    for counter, (x_batch, y_batch) in enumerate(cifar10_train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    top1_train_accuracy /= (counter + 1)
    avg_loss = total_loss / (counter + 1)  # Calculate average training loss

    top1_accuracy = 0
    top5_accuracy = 0
    train_loss_list.append(avg_loss)
    for counter, (x_batch, y_batch) in enumerate(cifar10_test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)

    top1_train_accuracy_list.append(top1_train_accuracy.item())
    top1_accuracy_list.append(top1_accuracy.item())
    top5_accuracy_list.append(top5_accuracy.item())
    epoch_list.append(epoch + 1)

    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

    writer.add_scalar('Loss/train', avg_loss, epoch + 1)  # Write training loss to TensorBoard
    writer.add_scalar('Accuracy/train_top1', top1_train_accuracy.item(), epoch + 1)  # Write training accuracy to TensorBoard
    writer.add_scalar('Accuracy/test_top1', top1_accuracy.item(), epoch + 1)  # Write test accuracy to TensorBoard
    writer.add_scalar('Accuracy/test_top5', top5_accuracy.item(), epoch + 1)  # Write test top5 accuracy to TensorBoard

writer.close()  # Close the TensorBoard writer

top1_train_accuracy_list.pop(0)
top1_accuracy_list.pop(0)
top5_accuracy_list.pop(0)
epoch_list.pop(0)

plt.figure(figsize=(16, 9))
plt.title('Training Loss Curve')
plt.plot(epoch_list, train_loss_list, c='m')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()



plt.figure(figsize=(16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR10 Accuracy Plot')
plt.plot(epoch_list, top1_train_accuracy_list, c='b')
plt.plot(epoch_list, top1_accuracy_list, c='g')
plt.plot(epoch_list, top5_accuracy_list, c='r')
plt.legend(['Training Accuracy', 'Top 1 Test Accuracy', 'Top 5 Test Accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
