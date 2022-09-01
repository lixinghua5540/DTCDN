import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F
class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class DoubleConvolutionLayer(nn.Module):
    """
    The DoubleConvolutionLayer is an module extending torch Module. It is a module that contains a layer that applies
    Conv2d -> Batchnorm -> ReLU -> Conv2d -> Batchnorm -> ReLU.
    It changes the number of channels of the input but not it's size.
    """
    def __init__(self, n_channels_input,n_middle_channel, n_channels_output):
        """
        Initialises a DoubleConvolutionLayer object containing one layer that procedes to the operations described
        above sequentially.
        :param n_channels_input: number of channels in input
        :param n_channels_output: number of channels in output
        """
        super(DoubleConvolutionLayer, self).__init__()
        self.double_layer = nn.Sequential(
                                          AtrousSeparableConvolution(in_channels=n_channels_input,out_channels=n_middle_channel,kernel_size=3,stride=1, padding=1, dilation=1, bias=True),
                                          #nn.Conv2d(n_channels_input, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_middle_channel),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout2d(p=0.05),
                                          AtrousSeparableConvolution(in_channels=n_middle_channel, out_channels=n_channels_output,kernel_size=3,stride=1, padding=1, dilation=1, bias=True),
                                          #nn.Conv2d(n_channels_output, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_channels_output),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout2d(p=0.05)
                                          )


    def forward(self, x):
        """
        Defines the flow of data in the DoubleConvolution object.
        :param x: the input data given through the layer with n_channels_input channels
        :return: x after passing through the layer with now n_channels_output channels.
        """
        x = self.double_layer(x)
        return x
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.ca = ChannelAttention(out_ch, ratio=8)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
       # output = self.ca(output)*output+output
        return output



class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class ExternalAttention(nn.Module):
    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class AdaptiveFF(nn.Module):
    def __init__(self, in_channels,ratio = 4):
        super(AdaptiveFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmod(avg_out)
        out = out*x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class Adaptive_ChannelAttention(nn.Module):
    def __init__(self, in_channels, S = 16 ,r = 8):
        super(Adaptive_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//r,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//r, in_channels,1,bias=False)
        self.fc3 = nn.Conv2d(in_channels*2, in_channels // S, 1, bias=False)
        self.fc4 = nn.Conv2d(in_channels//S, 2,1,bias=False)
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
    def forward(self,x):
        avg_out = self.sigmod(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))))
        max_out =  self.sigmod(self.fc2(self.relu1(self.fc1(self.max_pool(x)))))
        out1 = torch.cat([avg_out, max_out], 3)
        out2 = torch.cat([avg_out, max_out], 1)

        DA = self.softmax(self.fc4(self.relu1(self.fc3(out2))))
        DA =  DA.permute(0,2,3,1)
        out1 = out1.permute(0,2,3,1)
        out =torch.matmul(DA,out1)
        out = out.permute(0,3,1,2)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_window=6, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.SE = nn.Sequential(
           #nn.AvgPool2d(kernel_size=pool_window + 1, stride=1, padding=pool_window // 2),
            nn.Conv2d(in_channels*2, in_channels*2 // reduction, 1),
            nn.BatchNorm2d(in_channels*2 // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels*2 // reduction, out_channels, 1),
            nn.Sigmoid())

    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        avg_out = self.avg_pool(high_feat)
        max_out = self.max_pool(high_feat)
        high_feat = torch.cat([avg_out, max_out], dim=1)
        A = self.SE(high_feat)
        A = F.upsample(A, (h, w), mode='bilinear')

        output = low_feat * A
        if self.add_input:
            output += low_feat

        return output

class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.AE0_1 = Attention_Embedding(filters[1] * 2, filters[0]*2)
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.AE1_1 = Attention_Embedding(filters[2] * 2, filters[1] * 2)
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.AE2_1 = Attention_Embedding(filters[3] * 2, filters[2] * 2)
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.AE3_1 = Attention_Embedding(filters[4] * 2, filters[3] * 2)

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.AE0_2 = Attention_Embedding(filters[1], filters[0] * 3 )
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.AE1_2 = Attention_Embedding( filters[2], filters[1] * 3 )
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.AE2_2 = Attention_Embedding(filters[3], filters[2] * 3)

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.AE0_3 = Attention_Embedding(filters[1], filters[0] * 4 )
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.AE1_3 = Attention_Embedding(filters[2], filters[1] * 4  )

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])
        self.AE0_4 = Attention_Embedding( filters[1], filters[0] * 5)


        self.Up1_0 = up(filters[1])
        self.Up2_0 = up(filters[2])
        self.Up3_0 = up(filters[3])
        self.Up4_0 = up(filters[4])

        self.Up1_1 = up(filters[1])
        self.Up2_1 = up(filters[2])
        self.Up3_1 = up(filters[3])

        self.Up1_2 = up(filters[1])
        self.Up2_2 = up(filters[2])
        self.Up1_3 = up(filters[1])

        self.ca = ChannelAttention(filters[0] * 4, ratio=8)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,xB,xA):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        x4_0A = self.conv4_0(self.pool(x3_0A))

        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.AE0_1(torch.cat([x1_0A, x1_0B], 1), torch.cat([x0_0A, x0_0B], 1))
        x0_1 = self.conv0_1(torch.cat([x0_1, self.Up1_0(x1_0B)], 1))

        x1_1 = self.AE1_1(torch.cat([x2_0A, x2_0B], 1), torch.cat([x1_0A, x1_0B], 1))
        x1_1 = self.conv1_1(torch.cat([x1_1,self.Up2_0(x2_0B)], 1))

        x0_2 = self.AE0_2(x1_1, torch.cat([x0_0A, x0_0B, x0_1], 1))
        x0_2 = self.conv0_2(torch.cat([x0_2,self.Up1_1(x1_1)], 1))

        x2_1 = self.AE2_1(torch.cat([x3_0A, x3_0B], 1), torch.cat([x2_0A, x2_0B], 1))
        x2_1 = self.conv2_1(torch.cat([x2_1,self.Up3_0(x3_0B)], 1))

        x1_2 = self.AE1_2(x2_1, torch.cat([x1_0A, x1_0B, x1_1], 1))
        x1_2 = self.conv1_2(torch.cat([x1_2, self.Up2_1(x2_1)], 1))

        x0_3 = self.AE0_3(x1_2, torch.cat([x0_0A, x0_0B, x0_1, x0_2], 1))
        x0_3 = self.conv0_3(torch.cat([x0_3, self.Up1_2(x1_2)], 1))

        x3_1 = self.AE3_1(torch.cat([x4_0A, x4_0B], 1), torch.cat([x3_0A, x3_0B], 1))
        x3_1 = self.conv3_1(torch.cat([x3_1, self.Up4_0(x4_0B)], 1))
        x2_2 = self.AE2_2(x3_1, torch.cat([x2_0A, x2_0B, x2_1], 1))
        x2_2 = self.conv2_2(torch.cat([x2_2, self.Up3_1(x3_1)], 1))
        x1_3 = self.AE1_3(x2_2, torch.cat([x1_0A, x1_0B, x1_1, x1_2], 1))
        x1_3 = self.conv1_3(torch.cat([x1_3, self.Up2_2(x2_2)], 1))
        x0_4 = self.AE0_4(x1_3, torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3], 1))
        x0_4 = self.conv0_4(torch.cat([x0_4, self.Up1_3(x1_3)], 1))


        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        #out = sa*out+out
        out = self.conv_final(out)

        return out


class Siam_NestedUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
        return (output1, output2, output3, output4, output)