from functools import partial
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F




class EMAU(nn.Module):  # 该模块最后用了一个残差连接，将输入与该注意力模块的结果连起来了
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)

        #应该就是再内存中定义了一个常量，同时模型保存和加载的时候可以读出。
        # 也就是说不参与反向传播,只在forword中进行计算
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c,momentum=3e-4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x

        # The first 1x1 conv，用来将将值域从(0, 正无穷)转为 (负无穷, 正无穷)。
        # 不这样做的话，后面得到的μ(T )的值域也是（0, 正无穷）。
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))

                mu = torch.bmm(x, z_)  # b * c * k   M步,Likelihood Maximization
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n    这就是经过注意力后得到的特尔征图
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv  用于将˜X 映射到了X残差空间
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        #这就是上面提到的被移除到train.py中的代码(作者移到train.py是和多卡寻来你有关),
        # 该代码是平均移动滑步,用于更新下一个epoch的mu的初始值,以获得更稳定的训练

        if self.training:
            mu = mu.mean(dim=0, keepdim=True)
            em_mom=3e-4
            self.mu *= em_mom
            self.mu += mu * (1 - em_mom)



        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

