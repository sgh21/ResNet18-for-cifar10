'''
RsesNet-18 Image classfication for cifar-10 with PyTorch 


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.conv=nn.Sequential(
            # 3x3 two kernels without size change
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out=self.conv(x)
        out+=self.shortcut(x)
        out=F.relu(out)
        return out
    
class ResNet(nn.Module):
    # ResBlock：残差模块儿类型 num_classes:分类数
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.criteon=nn.CrossEntropyLoss()#自带了softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.inchannel = 64
        #x:[b,64,32,32]->[b,64,32,32]->[b,64,32,32]
        self.layer1 = self.make_layer(ResBlock, 64,  2, stride=1)
        #x:[b,64,32,32]->[b,128,16,16]->[b,128,16,16]
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        #x:[b,128,16,16]->[b,256,8,8]->[b,256,8,8]
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        #x:[b,256,8,8]->[b,512,4,4]->[b,512,4,4]
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        #x:[b,512,4,4]->[b,512,1,1]
        self.avePool=nn.AvgPool2d(kernel_size=4,stride=4)
        #x:[b,512*1*1]->[b,num_classes]
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avePool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def ResNet18():

    return ResNet(ResBlock)