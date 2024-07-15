import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 超参数
learning_rate = 1e-3 # 学习率
batch_size = 64 # 每次迭代的样本数
epochs = 5 # 迭代次数
flag_train = input("是否训练模型？(y/n): ") == 'y'
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
        x = self.flatten(x) # 将二维数据展平为一维
        logits = self.linear_relu_stack(x) # 根据前面定义的神经网络可以得到输出 logits
        return logits

# 得到训练和测试数据
def get_data_loader(is_train, batch_size):
    data = datasets.FashionMNIST(root="data",train=is_train,download=True,transform=ToTensor())
    return DataLoader(data, batch_size=batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train() # 进入训练模式
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward() # 根据损失函数计算梯度
        optimizer.step() # 梯度下降的实现
        optimizer.zero_grad() # 在进行下一次迭代前，需要清零已经累积的梯度
 
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"损失: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval() # 进入测试模式
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"测试结果: \n 准确率：: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")

def main():
    train_dataloader = get_data_loader(True, batch_size)
    test_dataloader = get_data_loader(False, batch_size)
    model = NeuralNetwork()

    # 初始化损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if flag_train:
        for t in range(epochs):
            print(f"轮次 {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        torch.save(model, 'Machine_Learning/Pytorch/Fassion_model.pth')
        print("训练完成！")
    else:
        model = torch.load('Machine_Learning/Pytorch/Fassion_model.pth')
        for (n, (x, y)) in enumerate(test_dataloader):
            if(n > 5):
                break
            predict = torch.argmax(model.forward(x[0].view(-1, 28*28)))
            plt.figure(n)
            plt.imshow(x[0].view(28, 28))
            plt.title("predict: " + labels_map[int(predict)] + " currrent: " + labels_map[y[0].item()])
        plt.show()

if __name__ == '__main__':
    main()