import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5,0.5))  # 标准化
])

# 加载YaleFace数据集
dataset = datasets.ImageFolder('./CroppedYale', transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)
'''
# 查看训练数据的维度
for images, labels in train_loader:
    print(images.shape)
    break  # 只打印一个批次的数据维度
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层（编码器部分）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 100)
        )

        # 反卷积层（解码器部分）
        self.decoder = nn.Sequential(
            nn.Linear(100, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
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
        x = x / x_norm.clamp(min=1.0) #恒模约束
        x = self.decoder(x)
        return x
model=Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001) #优化器
loss_function = nn.MSELoss().cuda()

def train():
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.cuda()
        optimizer.zero_grad()
        reconstructions = model(images)
        loss = loss_function(reconstructions, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

num_epochs = 10  # 设置训练次数
for epoch in range(num_epochs):
    train_loss = train()
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")


# 选取第一批次的第一张图片
image, _ = next(iter(train_loader))
image = image[0]

# 加入高斯噪声后放入模型
noise_power = 0.01
noise = noise_power * torch.randn_like(image)
noisy_image = image + noise
noisy_image = noisy_image.cuda()
reconstructed_image = model(noisy_image.unsqueeze(0)).squeeze(0)

"""""
# 增加结构噪声(中间)
noisy_image = image.clone()
noisy_image[:, 20:40, 20:40] = 0  # 在中间的区域设置为0或其他噪声值
noisy_image = noisy_image.cuda()
reconstructed_image = model(noisy_image.unsqueeze(0)).squeeze(0) #模型扩展，添加批次维度 [C,H,W] -> [bacth_size,C,H,W]
"""""
"""""
#添加结构性噪声(角落)
noisy_image = image.clone()
noisy_image[:, :20, :20] = 0  # 在左上角的区域设置为0或其他噪声值
noisy_image = noisy_image.cuda()
reconstructed_image = model(noisy_image.unsqueeze(0)).squeeze(0) #模型扩展，添加批次维度 [C,H,W] -> [bacth_size,C,H,W]
"""""
# 将Tensor数据转换为numpy数组
image = image.cpu().numpy()
noisy_image = noisy_image.cpu().numpy()
reconstructed_image = reconstructed_image.detach().cpu().numpy()

image = (image + 1) / 2
noisy_image = (noisy_image + 1) / 2
reconstructed_image = (reconstructed_image + 1) / 2

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ['Original Image', 'Model input (Noisy Image)', 'Model output (Reconstructed Image)']
images = [image, noisy_image, reconstructed_image]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_title(title)
    ax.axis('off')

plt.show()



