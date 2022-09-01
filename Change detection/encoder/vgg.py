import torch
from torch import nn
import numpy as np
import sys
sys.path.append(r'D:\SemanticSeg_Pytorch')
import core.configures as cfg1
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

"""

Copied from torchvision.model.vgg
Remove FC+VIEW
Remove pretrained model download from parameters
Add encoder lavers as return

"""



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]



class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features_0 = features[0]
        self.features_1 = features[1]
        self.features_2 = features[2]
        self.features_3 = features[3]
        self.features_4 = features[4]

    def forward(self, x):
        x = self.features_0(x)
        x0=x
        x = self.features_1(x)
        x1=x
        x = self.features_2(x)
        x2=x
        x = self.features_3(x)
        x3=x
        x = self.features_4(x)
        x4=x
        dic={'p0':x0,'p1':x1,'p2':x2,'p3':x3,'p4':x4}
        return x,dic



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    layers_seq=[]
    for v in cfg:
        if v == 'M':
            layers_seq.append(nn.Sequential(*layers))
            layers= [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:

                layers += [conv2d, nn.BatchNorm2d(v,eps=cfg1.BN_EPSILON,momentum=cfg1.BN_MOMENTUM), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers_seq.append(nn.Sequential(*layers))
    return layers_seq


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm,  **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model


def vgg11( **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False,  **kwargs)


def vgg11_bn( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False,  **kwargs)


def vgg13_bn( **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg16_bn( **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True,  **kwargs)


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False,  **kwargs)


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True,  **kwargs)
def test():
    img=np.random.rand(1,3,512,512)
    x=torch.tensor(img,dtype=torch.float32)
    model=vgg16_bn()
    a,features=model(x)
    a
# test()
