import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ResBlock import ResNet,ResBlock
device=torch.device('cuda')
def ModelTest():
    # 定义数据处理和归一化操作
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # 创建测试集的 DataLoader
    batch_size = 1
    test_dataset = torchvision.datasets.CIFAR10('cifar', train=True, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    # netModel = torch.load('./model_SGD_0.03-0.005_64_93.6/full_model.pth')  # 替换为你定义的模型
    netModel=ResNet(ResBlock).to(device)
    netModel.load_state_dict(torch.load('./model/model_weights.pth'))
    # 设置模型为评估模式
    netModel.eval()

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义正确预测的数量
    correct = 0
    total = 0
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 在测试集上进行验证
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            # print(inputs.shape,labels.shape)
            outputs = netModel(inputs)
            _, predicted = torch.max(outputs, 1)
            print(classes[predicted.item()],classes[labels],predicted == labels)
            if predicted != labels:
                tensor_img=inputs
                image_output(tensor_img)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
    print(total)
def image_output(tensor_img):
    #去除batch维度
    tensor_img=tensor_img.view(3,32,32)
    # 反归一化
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    tensor_img = tensor_img.permute(1, 2, 0).cpu().numpy()
    tensor_img = tensor_img*std+mean
    to_img=transforms.ToPILImage()
    img=to_img(tensor_img)
    img.show()
    input()
if __name__=="__main__":
    ModelTest()