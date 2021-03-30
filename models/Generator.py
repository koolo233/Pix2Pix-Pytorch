import torch
from torch import nn
from collections import OrderedDict


class EnConv(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, padding, activate=None, if_bn=True, if_dropout=False):
        """
        :param in_c: 输入通道数
        :param out_c: 输出通道数
        :param kernel_size: 卷积核size
        :param stride: 步长
        :param padding: 填充
        :param activate: 激活函数
        :param if_bn: 是否使用bn层
        :param if_dropout: 是否使用dropout
        """
        super(EnConv, self).__init__()

        self.conv = nn.Sequential()

        if activate is not None:
            if activate == "LeakyReLU":
                self.conv.add_module("leaky relu", nn.LeakyReLU(0.2, inplace=True))
            elif activate == "ReLU":
                self.conv.add_module("relu", nn.ReLU(inplace=True))

        self.conv.add_module("conv", nn.Conv2d(in_c, out_c, kernel_size, stride, padding))

        if if_bn:
            self.conv.add_module("bn", nn.BatchNorm2d(out_c))

        if if_dropout:
            self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, x):
        return self.conv(x)


class DecodeConv(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, padding, if_bn=True, activate=None, if_dropout=False):
        """
        :param in_c:
        :param out_c:
        :param kernel_size:
        :param stride:
        :param padding:
        :param if_bn:
        :param activate:
        :param if_dropout:
        """

        super(DecodeConv, self).__init__()

        self.conv = nn.Sequential()

        self.conv.add_module("trans conv", nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding))

        if if_bn:
            self.conv.add_module("bn", nn.BatchNorm2d(out_c))

        if if_dropout:
            self.conv.add_module("dropout", nn.Dropout(0.5))

        self.activate = activate

    def forward(self, x1, x2):
        if self.activate is not None:
            if self.activate == "ReLU":
                output = nn.ReLU(inplace=True)(x1)

        output = self.conv(output)
        output = torch.cat([output, x2], dim=1)
        return output


class Generator(nn.Module):

    def __init__(self, in_c, out_c, ngf):
        super(Generator, self).__init__()

        # 256 256
        self.encode_inconv = EnConv(in_c, ngf, kernel_size=4, stride=2, padding=1, activate=None, if_bn=False)
        # 128 128
        self.encode_conv1 = EnConv(ngf, ngf*2, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 64 64
        self.encode_conv2 = EnConv(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 32 32
        self.encode_conv3 = EnConv(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 16 16
        self.encode_conv4 = EnConv(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 8 8
        self.encode_conv5 = EnConv(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 4 4
        self.encode_conv6 = EnConv(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=True)
        # 2 2
        self.encode_conv7 = EnConv(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, activate="LeakyReLU", if_bn=False)
        # 1 1

        self.decode_conv1 = DecodeConv(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, activate="ReLU",
                                       if_bn=True, if_dropout=True)
        # 2 2
        self.decode_conv2 = DecodeConv(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1, activate="ReLU",
                                       if_bn=True, if_dropout=True)
        # 4 4
        self.decode_conv3 = DecodeConv(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1, activate="ReLU",
                                       if_bn=True, if_dropout=True)
        # 8 8
        self.decode_conv4 = DecodeConv(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1, activate="ReLU", if_bn=True)
        # 16 16
        self.decode_conv5 = DecodeConv(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1, activate="ReLU", if_bn=True)
        # 32 32
        self.decode_conv6 = DecodeConv(ngf*4*2, ngf*2, kernel_size=4, stride=2, padding=1, activate="ReLU", if_bn=True)
        # 64 64
        self.decode_conv7 = DecodeConv(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1, activate="ReLU", if_bn=True)
        # 128 128
        self.decode_output = nn.Sequential(OrderedDict([
            ("relu", nn.ReLU(inplace=True)),
            ("output conv", nn.ConvTranspose2d(ngf*2, out_c, 4, 2, 1)),
            ("output activate", nn.Tanh())
        ]))
        # 256 256

    def forward(self, x):
        e1 = self.encode_inconv(x)
        e2 = self.encode_conv1(e1)
        e3 = self.encode_conv2(e2)
        e4 = self.encode_conv3(e3)
        e5 = self.encode_conv4(e4)
        e6 = self.encode_conv5(e5)
        e7 = self.encode_conv6(e6)
        e8 = self.encode_conv7(e7)

        d1 = self.decode_conv1(e8, e7)
        d2 = self.decode_conv2(d1, e6)
        d3 = self.decode_conv3(d2, e5)
        d4 = self.decode_conv4(d3, e4)
        d5 = self.decode_conv5(d4, e3)
        d6 = self.decode_conv6(d5, e2)
        d7 = self.decode_conv7(d6, e1)
        output = self.decode_output(d7)
        return output

    def weights_init(self, mod):
        """设计初始化函数"""
        classname = mod.__class__.__name__
        if classname.find('Conv') != -1:  # 这里的Conv和BatchNnorm是torc.nn里的形式
            mod.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            mod.weight.data.normal_(1.0, 0.02)  # bn层里初始化γ，服从（1，0.02）的正态分布
            mod.bias.data.fill_(0)  # bn层里初始化β，默认为0


if __name__ == "__main__":
    # generator test
    generator = Generator(3, 3, 32)
    noise = torch.randn((1, 3, 256, 256), dtype=torch.float32)
    output = generator(noise)
    print(output.shape)
