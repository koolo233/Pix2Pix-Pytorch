# Pix2Pix-Pytorch
pix2pix的复现

## Pix2Pix介绍

1. cGAN能有效控制生成图像与原图的相似度，并有助于保持高频信息
2. L1 loss与L2 loss均会导致图像模糊，即保留低频信息，而忽略高频信息
3. PatchGAN一方面能有效降低参数量，另一方面能有效的监督高频信息
4. UNet的跳连结构能有效降低bottleneck对信息流动的影响

### 网络结构
Ck：具有k个卷积核的Conv-BN-ReLU结构
CDk：具有k个卷积核的TransConv-BN-Dropout-ReLU结构

所有的卷积核为4x4，stride为2

#### 生成器：
编码部分：
C64-C128-C256-C512-C512-C512-C512-C512
解码部分：
CD512-CD512-CD512-C512-C256-C128-C64


#### U-Net结构下的解码部分：
CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128


输出：使用一个等同于目标通道数的卷积，并使用Tanh激活
特别说明：第一个卷积不使用BN，所有的编码ReLU使用LeakyReLU，斜率为0.2，解码ReLU使用ReLU

#### 判别器：
70x70判别器：
C64-C128-C256-C512，最后接上一个卷积输出到单通道，使用Sigmoid激活，第一个卷积不使用BN，使用斜率为0.2的LeakyReLU
1x1判别器：
C64-C128，使用1x1的卷积核
16x16判别器：
C64-C128
286x286判别器：
C64-C128-C256-C512-C512-C512

### 损失函数
两部分：GAN损失、生成图与目标原图的l1 loss  后者的权重为100

## 本项目目录结构

.
├── conf
│   └── pix2pix_settings.py  本项目参数设定
├── LICENSE
├── models
│   ├── Discriminator.py  判别器
│   └── Generator.py  生成器
├── README.md
├── train.py  训练文件
└── utils
    ├── CelebA_dataloader.py  CelebA数据集加载文件
    ├── Cifar10_dataloader.py  Cifar10数据集加载文件
    ├── DataLoaders.py  数据集加载函数管理
    ├── edge2shoes_dataloader.py  edge2shoes数据集加载文件 
    ├── mnist_dataloader.py  mnist数据集加载文件
    └── Mogaoku_dataloader.py  莫高窟壁画加载文件
