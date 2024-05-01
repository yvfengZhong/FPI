from torch import nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock

class VGG16(nn.Module):
    def __init__(self, cfg):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.conv_body = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x):
        output = []
        output.append(self.conv_body(x))
        return output


class ResNet50(nn.Module):
    def __init__(self, cfg, flag):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv_body = nn.Sequential(*list(resnet.children())[:int(flag - 7)])

    def forward(self, x):
        output = []
        output.append(self.conv_body(x))
        return output


class ResNet18(ResNet):
    def __init__(self):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x