import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append(r'D:\SemanticSeg_Pytorch')
import core.configures as cfg
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip=nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride),nn.Conv2d(inplanes, planes, 1, stride=1, bias=False))
            # self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        sep_num=0
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True))
            filters = planes
            sep_num=sep_num+1

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True))
            sep_num=sep_num+1
            if sep_num==2:
                self.rep1 = nn.Sequential(*rep)
                rep=[]

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True))
            sep_num=sep_num+1
            if sep_num==2:
                self.rep1 = nn.Sequential(*rep)
                rep=[]
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True))

        if not start_with_relu:
            self.rep1 = self.rep1[1:]

        self.rep2 = nn.Sequential(*rep)

    def initialize_weights(self):
        if self.skip is not None:
            self.skipbn.weight.data.fill_(0)
            self.skipbn.bias.data.zero_()

    def forward(self, inp):
        x = self.rep1(inp)
        p=x
        x = self.rep2(x)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x,p
class Xception41(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride=16, BatchNorm=nn.BatchNorm2d,
                 pretrained=True):
        super(Xception41, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048,eps=cfg.BN_EPSILON,momentum=cfg.BN_MOMENTUM,track_running_stats=True)
    def initialize_weights(self):
        print('Initializing BN in Res Block!')
        self.block1.initialize_weights()
        self.block2.initialize_weights()
        self.block3.initialize_weights()
        self.block4.initialize_weights()
        self.block5.initialize_weights()
        self.block6.initialize_weights()
        self.block7.initialize_weights()
        self.block8.initialize_weights()
        self.block9.initialize_weights()
        self.block10.initialize_weights()
        self.block11.initialize_weights()
        self.block12.initialize_weights()
        self.block13.initialize_weights()
        self.block14.initialize_weights()
        self.block15.initialize_weights()
        self.block16.initialize_weights()
        self.block17.initialize_weights()
        self.block18.initialize_weights()
        self.block19.initialize_weights()
        self.block20.initialize_weights()
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
#32,256,256
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x,p1= self.block1(x)#p1:128,256,256
        # add relu here
        x = self.relu(x)

        #128,128,128
        x,p2 = self.block2(x)#p2:256,128,128
        #256,64,64
        x,p3= self.block3(x)#p3:728,64,64
        # Middle flow
        x,_ = self.block4(x)
        x,_ = self.block5(x)
        x,_ = self.block6(x)
        x ,_= self.block7(x)
        x,_ = self.block8(x)
        x,_ = self.block9(x)
        x,_ = self.block10(x)
        x,_ = self.block11(x)
        x,_ = self.block12(x)
        x,_ = self.block13(x)
        x,_ = self.block14(x)
        x,_ = self.block15(x)
        x,_ = self.block16(x)
        x,_ = self.block17(x)
        x,_ = self.block18(x)
        x,_ = self.block19(x)

        # Exit flow
        x,_ = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        dic={'p1':p1,'p2':p2,'p3':p3,'p4':x}
        return x, dic
def test():
    img=np.random.rand(1,3,512,512)
    x=torch.tensor(img,dtype=torch.float32).cuda()
    a=Xception41().cuda()
    b,feature=a(x)
    b
# test()