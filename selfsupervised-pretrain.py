"""
    导入必备的库，下载和预处理Cifar10数据集

"""
import os
import shutil
import torch
import yaml
from torchvision.transforms import transforms
from torchvision import transforms, datasets
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import torch.nn as nn
import torchvision.models as models
import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import Namespace
import torch

np.random.seed(0)


"""
    数据集的相关处理

"""

class ViewGen(object):
    # Take 2 random crops

    def __init__(self, base_transform, n_views=2):
        # 构造函数，接受一个基础转换函数和生成视图的数量
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # 可调用方法，接受输入x，并生成多个视图的列表
        return [self.base_transform(x) for i in range(self.n_views)]



# GetTransformedDataset类用于获取经过转换的数据集

class GetTransformedDataset:
    @staticmethod
    def get_simclr_transform(size, s=1):
        # 获取SimCLR的转换器
        # size: 图像的尺寸
        # s: 颜色抖动的幅度
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomApply([color_jitter], p=0.8),  # 以0.8的概率应用颜色抖动
            transforms.RandomGrayscale(p=0.2),  # 以0.2的概率转换为灰度图像
            GaussianBlur(kernel_size=int(0.1 * size)),  # 高斯模糊
            transforms.ToTensor()  # 转换为张量
        ])
        return data_transforms

    def get_cifar10_train(self, n_views):
        # 获取CIFAR-10训练集
        # n_views: 生成视图的数量
        return datasets.CIFAR10('.', train=True,
                                transform=ViewGen(
                                    self.get_simclr_transform(32),
                                    n_views),
                                download=True)

    def get_cifar10_test(self, n_views):
        # 获取CIFAR-10测试集
        # n_views: 生成视图的数量
        return datasets.CIFAR10('.', train=False,
                                transform=transforms.ToTensor(),
                                download=True)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    保存模型检查点文件

    参数:
        state: 包含模型状态和其他信息的字典
        is_best: 布尔值，表示当前模型是否是最佳模型
        filename: 检查点文件的名称，默认为'checkpoint.pth.tar'
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    """
    保存配置文件

    参数:
        model_checkpoints_folder: 模型检查点文件夹的路径
        args: 包含配置信息的字典或对象
    """
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """
    计算给定topk值的预测准确率

    参数:
        output: 模型的输出
        target: 真实标签
        topk: 一个整数或整数元组，表示要计算的topk值，默认为1

    返回值:
        返回一个包含topk准确率的列表
    """
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


np.random.seed(0)


# 添加高斯模糊
class GaussianBlur(object):
    def __init__(self, kernel_size):
        """
        初始化函数，接受一个参数 kernel_size，表示高斯核的大小。

        参数:
        kernel_size (int): 高斯核的大小

        属性:
        blur_h (nn.Conv2d): 水平方向的卷积层
        blur_v (nn.Conv2d): 垂直方向的卷积层
        k (int): 高斯核的大小
        r (int): 高斯核半径
        blur (nn.Sequential): 高斯模糊的序列操作
        pil_to_tensor (transforms.ToTensor): PIL 图像转换为张量的转换器
        tensor_to_pil (transforms.ToPILImage): 张量转换为 PIL 图像的转换器
        """
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


"""

         ResNet18的实现

"""
class get_resnet18(nn.Module):

    def __init__(self, out_dim):
        super(get_resnet18, self).__init__()
        self.backbone = self._get_basemodel(out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, out_dim):
        return models.resnet18(pretrained=False, num_classes=out_dim)

    def forward(self, x):
        return self.backbone(x)


torch.manual_seed(0)


"""

         自监督对比学习的实现------------预训练部分

"""
class simclr_framework(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(
            self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size)
                            for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
            self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar(
                        'acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar(
                        'acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[
                        0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': 'resnet18',
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


def main():
    args = Namespace
    args.batch_size = 500
    args.device = torch.device('cuda')
    args.disable_cuda = False
    args.epochs = 50
    args.fp16_precision = False
    args.gpu_index = -1
    args.log_every_n_steps = 1
    args.lr = 3e-4
    args.n_views = 2
    args.out_dim = 128
    args.seed = 1
    args.temperature = 0.07
    args.weight_decay = 0.0008
    args.workers = 4

    dataset = GetTransformedDataset()
    print('dataset loaded..')
    train_dataset = dataset.get_cifar10_train(args.n_views)
    print('train_dataset loaded..')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    print('train_loader loaded..')
    model = get_resnet18(out_dim=args.out_dim)
    print('model loaded..')
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay)
    print('optimizer loaded..')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    print('scheduler loaded..')
    simclr = simclr_framework(model=model, optimizer=optimizer,
                              scheduler=scheduler, args=args)
    print('simlcr loaded..')
    print('training started..')
    simclr.train(train_loader)
    print('training completed..')


if __name__ == "__main__":
    main()



