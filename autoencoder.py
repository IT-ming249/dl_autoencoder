import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 载入MNIST数据，训练集
train_dataset = datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_dataset = datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)

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
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), #卷积核为3*3 步长为2，padding在特征图外围补1圈0 黑白图像的通道数是1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4 ,100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100,64*4*4),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #noise=torch.randn_like(encoded)
        #encoded=encoded+noise  #在重构前加入噪声
        # 功率限制函数
        encoded_norm = torch.norm(encoded, dim=1, keepdim=True)
        #计算编码后的特征向量的范数。这里使用了torch.norm函数，其中dim=1表示计算每个特征向量的范数，keepdim=True表示保持维度数目不变，使得结果仍然是一个列向量。
        encoded = encoded / encoded_norm.clamp(min=1.0)
        #encoded_norm.clamp(min=1.0)：对计算得到的范数进行裁剪，将小于1的值设为1，以确保最小范数为1
        #encoded = encoded / encoded_norm.clamp(min=1.0)：将编码后的特征向量除以裁剪后的范数，实现单位范数化。
        # 这样做可以确保输出信号的功率限制为1，因为单位范数的向量具有固定的模长。  闲置功率相当于恒模约束
        noise=0.1*torch.randn_like(encoded)  #噪声功率可改
        encoded=encoded+noise #添加噪声

        decoded = self.decoder(encoded)
        decoded = decoded.view(decoded.size(0), 1, 28, 28)
        return decoded




net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001) #优化器
criterion = nn.MSELoss()#代价函数


def train(net, train_loader, optimizer, criterion):
    net.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.float()
        optimizer.zero_grad()
        reconstructions = net(images)
        loss = criterion(reconstructions, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

num_epochs = 5 #设置训练次数
for epoch in range(num_epochs):
    train_loss = train(net, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")


num_samples =8  # 要显示的样本数量
# 选择一个批次的图像
sample_images = inputs[:num_samples]  # 选择前num_samples张图像
# 将图像传递给网络获取重建结果
reconstructed_images = net(sample_images)
# 将图像转换为NumPy数组并还原像素范围
sample_images = sample_images.detach().numpy() * 255
reconstructed_images = reconstructed_images.detach().numpy() * 255
# 创建包含8个子图的图像
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# 在每个子图上绘制原始图像和重建图像
for i in range(num_samples):
    row = i % 4
    col = i // 4
    # 绘制原始图像
    axes[row, col * 2].imshow(sample_images[i][0], cmap='gray')
    axes[row, col * 2].set_title('Original')
    axes[row, col * 2].axis('off')
    # 绘制重建图像
    axes[row, col * 2 + 1].imshow(reconstructed_images[i][0], cmap='gray')
    axes[row, col * 2 + 1].set_title('Reconstructed')
    axes[row, col * 2 + 1].axis('off')

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图像
