import mindspore as ms
import mindspore.nn as nn  
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer, HeNormal

from mindspore import ms_function

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class LambdaLayer(nn.Cell):
    def __init__(self, padding):
        super(LambdaLayer, self).__init__()
        self.padding = padding
        
    def construct(self, x):
        return ops.pad(x[:, :, ::2, ::2], self.padding)

class BasicBlock(nn.Cell):
    expansion = 1  # 最后一个卷积核数量与第一个卷积核数量相等

    def __init__(self, in_channel, out_channel, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.shortcut = nn.SequentialCell()

        if stride != 1 or in_channel != out_channel:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                paddings = ((0, 0), (out_channel//4, out_channel//4), (0, 0), (0, 0))
                self.shortcut = LambdaLayer(paddings)
                
            elif option == 'B':
                self.shortcut = nn.SequentialCell([
                        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, has_bias=False),
                        nn.BatchNorm2d(out_channel)])
            
        
    def construct(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out
        

class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Dense(64, num_classes)
        self.reshape = ops.Reshape()


    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel

        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # pool = nn.AvgPool2d(out.shape[3],out.shape[3])
        # out = pool(out)
        out = ops.avg_pool2d(out, out.shape[3], out.shape[3])
        out = self.reshape(out, (out.shape[0], -1))
        out_last = self.linear(out)
        return out_last, out
        #return out



def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)



