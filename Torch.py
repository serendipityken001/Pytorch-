import torch
import numpy as np

# 1. 初始化张量
data = [[1, 2], [3, 4]]

# 1.1 直接来自于数据
x_data = torch.tensor(data)

# 1.2 从numpy数组转换
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 1.3 从另一个张量（不指定则保留原来的属性）
x_ones = torch.ones_like(x_data) # 保留x_data的属性
print(f"全为1的张量: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # 重写x_data的数据类型
print(f"随机数的张量: \n {x_rand} \n")

# 1.4 通过指定形状创建张量
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"随机数的张量: \n {rand_tensor} \n")
print(f"全为1的张量: \n {ones_tensor} \n")
print(f"全为0的张量: \n {zeros_tensor}")

# 2. 张量属性
tensor = torch.rand(3,4)
print(f"张量的维度: {tensor.shape}")
print(f"张量的类型: {tensor.dtype}")
print(f"设备张量存储位置: {tensor.device}")

# 3. 张量运算
if torch.cuda.is_available():
    tensor = tensor.to('cuda') 
    print(f"设备张量存储位置: {tensor.device}")

# 3.1 索引和切片
    tensor = torch.rand(4, 4)
    print('Last column:', tensor[..., -1])
    print('Last column:', tensor[:, -1])