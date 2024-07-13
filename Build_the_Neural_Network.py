# 构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 获取用于训练的设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"使用 {device} 设备\n")

# 继承 nn.Module 类，创建神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
X = torch.rand(1, 28, 28, device=device) # 生成一个随机张量，可以被视为一个28x28的单通道图像
logits = model(X) # 指未归一化的预测或者模型的最终线性层的输出
"""
对logits应用Softmax函数，将其转换为概率分布
dim=0 对应于每个类的 10 个原始预测值的每个输出
dim=1 对应于每个输出的单个值
"""
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"预测结果: {y_pred}")

input_image = torch.rand(3,28,28) # 三个28x28的单通道图像
print(input_image.size())

# 将输入张量展平成一维张量
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 创建线性层，输入特征数为28x28，输出特征数为20
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"应用ReLU激活函数前的张量值: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"应用ReLU激活函数后的张量值: {hidden1}")

# 包含四个层的容器
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image) # 通过向前传递输入来调用模型

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print(f"模型结构: {model}\n\n")

# 打印每个参数的名称、大小和前两个值
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

