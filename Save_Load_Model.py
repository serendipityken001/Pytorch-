import torch
import torchvision.models as models

# 1. 保存和加载模型的状态字典（state_dict）
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 2. 保存和加载整个模型
torch.save(model, 'model.pth')

model = torch.load('model.pth')

