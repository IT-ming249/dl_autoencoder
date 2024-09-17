import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 载入CIFAR10数据，训练集
train_dataset = datasets.CIFAR10(root='./', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 使用均值和标准差进行归一化
]), download=True)

# 测试集
test_dataset = datasets.CIFAR10(root='./', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 使用均值和标准差进行归一化
]), download=True)

# 设定批次大小
b_size = 64
# 装载训练集
train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
# 装载测试集
test_loader = DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=True)

for i,data in enumerate(train_loader): #enumerate()是一个 Python 内置函数，用于同时获取序列（如列表、元组、字符串等）的索引和元素值。这里面i代表批次索引
    inputs,labels=data#data代表每个批次的数据和标签
    print(inputs.shape) #64为批次大小，1代表黑白图像通道数(彩色为3)，28*28表示图片的大小
    print(labels.shape)
    break #这个循环只是查看一下训练集中的数据属性，所以只打印了一个批次


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 100)
        )

        # 反卷积层（解码器部分）
        self.decoder = nn.Sequential(
            nn.Linear(100, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x = x / x_norm.clamp(min=1.0)
        #noise = torch.randn_like(x)  # 噪声功率可改
        #x = x + noise  # 添加噪声
        x = self.decoder(x)
        return x



net = Net().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001) #优化器
criterion = nn.MSELoss().cuda()


def train(net, train_loader, optimizer, criterion):
    net.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.cuda()
        optimizer.zero_grad()
        reconstructions = net(images)
        loss = criterion(reconstructions, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

num_epochs = 20 #设置训练次数
for epoch in range(num_epochs):
    train_loss = train(net, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")

def visualize_images(images, reconstructions):
    num_images = min(images.size(0), 8)
    images = images.detach().cpu()
    reconstructions = reconstructions.detach().cpu()
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    for i in range(num_images):
        input_image = images[i].permute(1, 2, 0)
        input_image = ((input_image + 1) / 2 * 255).numpy().astype(np.uint8)
        axes[i, 0].imshow(input_image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Input Image')
        reconstructed_image = reconstructions[i].permute(1, 2, 0)
        reconstructed_image = ((reconstructed_image + 1) / 2 * 255).numpy().astype(np.uint8)
        axes[i, 1].imshow(reconstructed_image)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Reconstructed Image')
    plt.tight_layout()
    plt.show()

# 可视化输入图像和模型输出的图像
with torch.no_grad():
    for images, _ in train_loader:
        images = images.cuda()
        reconstructions = net(images)
        visualize_images(images, reconstructions)
        break  # 只可视化一个批次的图像



