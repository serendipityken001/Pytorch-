import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 对于训练，我们需要将特征作为归一化张量，并将标签作为单热编码张量。 为了进行这些转换，我们使用 ToTensor 和 Lambda
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # 创建独热编码
)