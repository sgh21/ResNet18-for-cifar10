import torch  
from torch.utils.data import DataLoader,random_split  #数据批量加载，随机切分
from torchvision import datasets #数据下载与定义
from torchvision import transforms #图片转换
from torch import nn
from torch.nn import functional as F
from ResBlock import ResNet18
import matplotlib.pyplot as plt

# 超参数设置
epochs=128   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
batch_size=128     #批处理尺寸(batch_size)
# learningRate = 5e-4       #学习率 Adam
learningRate = 3e-2       #学习率 SGD
#scheduler
T_max=128
eta_min=5e-3
# 整理数据
X = []
Y = []
def TrainAndTest():
    # 加载训练集
    cifar_train=datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),download=True)#单张图片加载模式
    cifar_train=DataLoader(cifar_train,batch_size=batch_size,shuffle=True)#批量加载
    # 加载测试集
    cifar_test=datasets.CIFAR10('cifar',train=False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),download=True)#单张图片加载模式
    cifar_test=DataLoader(cifar_test,batch_size=batch_size,shuffle=True)#批量加载

    #test loader
    # x,label=iter(cifar_train).__next__()
    # print('x:',x.shape,'label:',label.shape)
    device = torch.device('cuda')
    net=ResNet18().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # optimizer=torch.optim.Adam(net.parameters(),lr=learningRate,betas=(0.90,0.999))
    # Set up  learning rate scheduler
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=T_max,eta_min=eta_min)
    for epoch in range(epochs):
        net.train()
        for x,label in cifar_train:
            #x:[N,3,32,32],label:[N]
            x,label=x.to(device),label.to(device)#数据转移到GPU
            logits=net(x)
            loss=net.criteon(logits,label)
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #更新学习率
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])
        print(epoch,loss.item())
        net.eval()
        with torch.no_grad():
            #test 
            test_correct = 0
            total_num = 0
            for x,label in cifar_test:
                x,label=x.to(device),label.to(device)#数据转移到GPU
                # logits:[N,10]
                logits=net(x)
                pred=logits.argmax(dim=1)
                test_correct+=torch.eq(pred,label).float().sum().item()
                total_num+=x.size(0)
            acc=test_correct/total_num
            X.append(epoch)
            Y.append(acc)
            print(epoch,acc)
    torch.save(net.state_dict(),'model_weight.pth')
    torch.save(net,'full_model.pth')
    
if __name__=="__main__":
    TrainAndTest()
    # 绘制折线图
    plt.plot(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Accuracy-Bitch')
    plt.show()