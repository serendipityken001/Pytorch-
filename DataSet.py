import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader

"""
root 存储训练/测试数据的路径
train 指定训练或测试数据集
download 如果数据在 root 中不可用，则从 Internet 下载数据
transform 指定特征和标签转换
"""
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8)) # 画布大小
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    """
    torch.randint生成一个介于0和len(training_data)之间的随机整数
    size=(1,)指定生成一个数
    .item()将其转换为Python的标准整数
    """
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off") # 关闭子图的坐标轴，使图像更加清晰
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

"""
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
"""

"""
training_data 包含图像和标签的数据集
batch_size=64 意味着每个批次包含64个样本
shuffle=True 表示在每个epoch开始时数据将被打乱(这有助于模型学习泛化)
"""
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

"""
从训练数据的DataLoader中获取第一个批次的数据
iter(train_dataloader) 创建了一个迭代器
next()函数从迭代器中获取第一个元素(包含特征（图像）和标签的批次)
"""
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
plt.title(labels_map[label.item()])
plt.imshow(img, cmap="gray")
plt.show()