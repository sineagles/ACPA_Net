import torch
from torch import nn
from torch.nn.parameter import Parameter

from Attention.EMAU import EMAU
from Attention.cbam import SpatialGate, BasicConv, ChannelPool
import torch.nn.functional as F

class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3,Return_wieght=False):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.rw=Return_wieght

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class parallel_dailted_eca_layer(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,res=False,Return_wieght=False):
        super(parallel_dailted_eca_layer, self).__init__()
        self.cat=cat
        self.res=res
        self.rw=Return_wieght
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        strid=1
        rates = [strid * r for r in rates]
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.features=[]
        self.features.append(
             nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                   )
        for r in rates:
            self.features.append(
                nn.Conv1d(1, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                            )
        self.features = nn.ModuleList(self.features)
        if self.cat:
            self.fusion_cat=nn.Conv1d(4,1, kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  #y的shape是【[b, c, 1，1]

        # Two different branches of ECA module
        # 一维卷积是作用在做i后一个通道上的，.squeeze(-1)用于去掉最后一个为1的维度，使用.transpose(-1, -2)用于交换最后两个维度
        #此时y的维度变为了【b，1，c】，然后将一维卷积作用在y上面后再还原
        y=y.squeeze(-1).transpose(-1, -2)
        out=None
        # Multi-scale information fusion  到底是直接相加比较好还是通道连接比较好呢
        if self.cat:

            for f in self.features:
                if out == None:
                    out = f(y)
                else:
                    Y=f(y)
                    out =torch.cat((out,y),1)
            out=self.fusion_cat(out)
        else:   #不用通道连接，直接相加
            for f in self.features:
                if out==None:
                    out=f(y)
                else:
                    out=out+f(y)

        # 到底是用通道连接还是直接相加呢
        out=out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)

        if self.res:
            y = self.relu(x+x * y.expand_as(x))
            return y
        elif self.rw:
            return y
        else:
            return x * y.expand_as(x)

class parallel_dailted_eca_layer_r2(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False):
        super(parallel_dailted_eca_layer_r2, self).__init__()
        self.eca_layer1 = parallel_dailted_eca_layer(channel, strid=1,k_size=k_size, rates=rates, cat=cat)
        self.eca_layer2 = parallel_dailted_eca_layer(channel, strid, k_size=k_size,rates=rates, cat=cat)
    def forward(self,x) :
        sc=self.eca_layer1(x)
        sc=self.eca_layer2(sc)
        return sc


class  cascade_dailted_eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],bn=False,res=False):
        super(cascade_dailted_eca_layer, self).__init__()
        self.bn=bn
        self.res=res
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        strid=1
        rates = [strid * r for r in rates]
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.features=[]
        if self.bn:
            self.features.append(
                nn.Sequential( nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                               nn.BatchNorm2d(channel) ,
                               nn.ReLU(inplace=True)
                               )
            )

            for r in rates:
                self.features.append(nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, dilation=r, padding=r, bias=False),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True)
                )
                )
        else:
            self.features.append(
                 nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                       )

            for r in rates:
                self.features.append(
                    nn.Conv1d(1, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                                )

        self.features = nn.ModuleList(self.features)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y=y.squeeze(-1).transpose(-1, -2)
        out=None
        # Multi-scale information fusion
        for f in self.features:
            if out==None:
                out=f(y)
            else:
                out=f(out)

        out=out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)

        if self.res:
            y = self.relu(x+x * y.expand_as(x))
            return y
        else:

            return x * y.expand_as(x)

class new_dailted_eca_layer(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,pool_types=['avg', 'max'],res=False):
        super(new_dailted_eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_type=pool_types
        self.cat=cat
        self.res=res

        strid=1
        rates = [strid * r for r in rates]

        self.features=[]

        self.features.append(
             nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                   )
        #other rate
        for r in rates:
            self.features.append(
                nn.Conv1d(1, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                            )

        self.features = nn.ModuleList(self.features)
        if self.cat:
            self.fusion_cat1=nn.Conv1d(4,1, kernel_size=1,bias=False)
            self.fusion_cat2 = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x) :

        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        avg_y = self.avg_pool(x)
        max_y = self.max_pool(x)

        # Two different branches of ECA module
        avg_y= avg_y.squeeze(-1).transpose(-1, -2)
        max_y = max_y.squeeze(-1).transpose(-1, -2)
        out = None
        avg_out=None
        max_out=None
        for type in self.pool_type:
        # Multi-scale information fusion
            if type=='avg':
                if self.cat:
                    for f in self.features:
                        if avg_out == None:
                            avg_out = f(avg_y)
                        else:
                            avg_out = torch.cat(avg_out, f(avg_y))
                    avg_out = self.fusion_cat1(avg_out)

                else:
                    for f in self.features:
                        if avg_out == None:
                            avg_out = f(max_y)
                        else:
                            avg_out = avg_out + f(max_y)
                out=avg_out
            elif type=='max':
                if self.cat:
                    for f in self.features:
                        if max_out == None:
                            max_out = f(max_y)
                        else:
                            max_out = torch.cat(out, f(max_y))
                    max_out = self.fusion_cat2(max_out)
                else:
                    for f in self.features:
                        if max_out == None:
                            max_out = f(max_y)
                        else:
                            max_out = max_out + f(max_y)
                if out==None:
                    out=max_out
                else:
                    out=out+max_out

        out = out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)

        if self.res:
            y = self.relu(x + x * y.expand_as(x))
            return y

        else:
            return x * y.expand_as(x)


class new_dailted_eca_layer_cbam(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,pool_types=['avg', 'max'],res=False):
        super(new_dailted_eca_layer_cbam, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_type=pool_types
        self.cat=cat
        self.res=res
        rates = [strid * r for r in rates]

        self.features=[]

        self.features.append(
             nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                   )
        #other rate
        for r in rates:
            self.features.append(
                nn.Conv1d(1, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                            )
        self.features = nn.ModuleList(self.features)
        if self.cat:
            self.fusion_cat1=nn.Conv1d(4,1, kernel_size=1,bias=False)
            self.fusion_cat2 = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x) :
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        avg_y = self.avg_pool(x)
        max_y = self.max_pool(x)

        # Two different branches of ECA module
        max_y = max_y.squeeze(-1).transpose(-1, -2)
        out = None
        avg_out=None
        max_out=None
        for type in self.pool_type:
        # Multi-scale information fusion
            if type=='avg':
                if self.cat:
                    for f in self.features:
                        if avg_out == None:
                            avg_out = f(avg_y)
                        else:
                            avg_out = torch.cat(avg_out, f(avg_y))
                    avg_out = self.fusion_cat1(avg_out)

                else:
                    for f in self.features:
                        if avg_out == None:
                            avg_out = f(max_y)
                        else:
                            avg_out = avg_out + f(max_y)
                out=avg_out
            elif type=='max':
                if self.cat:
                    for f in self.features:
                        if max_out == None:
                            max_out = f(max_y)
                        else:
                            max_out = torch.cat(out, f(max_y))
                    max_out = self.fusion_cat2(max_out)
                else:
                    for f in self.features:
                        if max_out == None:
                            max_out = f(max_y)
                        else:
                            max_out = max_out + f(max_y)
                if out==None:
                    out=max_out
                else:
                    out=out+max_out

        out = out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)


        if self.res:
            y = self.relu(x + x * y.expand_as(x))
            return y
        else:

            return x * y.expand_as(x)
#
class new_dailted_eca_layer2(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,pool_types=['avg', 'max'],res=False):
        super(new_dailted_eca_layer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_type=pool_types
        self.cat=cat
        self.res=res
        rates = [strid * r for r in rates]

        self.features=[]

        self.features.append(
             nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                   )
        #other rate
        for r in rates:
            self.features.append(
                nn.Conv1d(2, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                            )
        self.features = nn.ModuleList(self.features)
        if self.cat:
            self.fusion_cat1=nn.Conv1d(4,1, kernel_size=1,bias=False)
            self.fusion_cat2 = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x) :

        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        avg_y = self.avg_pool(x)
        max_y = self.max_pool(x)

        # Two different branches of ECA module

        avg_y= avg_y.squeeze(-1).transpose(-1, -2)
        max_y = max_y.squeeze(-1).transpose(-1, -2)
        y=torch.cat((avg_y,max_y),1)
        out=None
        if self.cat:
            for f in self.features:
                if out == None:
                    out = f(y)
                else:
                    out = torch.cat(out, f(y))
            out = self.fusion_cat(out)
        else:
            for f in self.features:
                if out == None:
                    out = f(y)
                else:
                    out = out + f(y)

        out = out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)
        if self.res:
            y = self.relu(x + x * y.expand_as(x))
            return y
        else:

            return x * y.expand_as(x)


#
class new_dailted_eca_layer2_cbam(nn.Module):
    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,pool_types=['avg', 'max'],res=False):
        super(new_dailted_eca_layer2_cbam, self).__init__()
        self.new_dailted_eca_layer2=new_dailted_eca_layer2( channel, strid,k_size,rates,cat,pool_types,res)
        self.SG = SpatialGate()
    def forward(self,x) :

        y=self.new_dailted_eca_layer2(x)
        y=self.SG(y)
        return y

class new_dailted_eca_layer2_dailted_cbam(nn.Module):

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,pool_types=['avg', 'max'],res=False):
        super(new_dailted_eca_layer2_dailted_cbam, self).__init__()
        self.new_dailted_eca_layer2=new_dailted_eca_layer2( channel, strid,k_size,rates,cat,pool_types,res)
        self.SG=dailted_SpatialGate(strid,rates)
    def forward(self,x) :

        y=self.new_dailted_eca_layer2(x)
        y=self.SG(y)
        return y

class dailted_eca_With_emau(nn.Module):
    def __init__(self, channel, strid, emau_k,k_size=3, rates=[2, 3, 4], cat=False, res=False,mode=False):

        super(dailted_eca_With_emau, self).__init__()
        self.mode=mode
        self.res=res
        self.EMAU=EMAU(channel, emau_k)
        self.eca_layer=parallel_dailted_eca_layer(channel, strid,rates=rates,cat=cat)
    def forward(self,x) :
        sc=self.eca_layer(x)
        if self.mode=="parallel":

            sa,_=self.EMAU(x)
            out=sc+sa
        else:

            out,_=self.EMAU(sc)
        if self.res:
            return x+out
        else:
            return out

class dailted_eca_With_cbam(nn.Module):
    def __init__(self, channel, strid,k_size=3, rates=[2, 3, 4], cat=False, res=False):
        super(dailted_eca_With_cbam, self).__init__()
        self.res=res
        self.SG=SpatialGate()
        self.eca_layer=parallel_dailted_eca_layer(channel, strid,k_size=k_size,rates=rates,cat=cat)
    def forward(self,x) :
        sc=self.eca_layer(x)
        out=self.SG(sc)
        if self.res:
            return x+out
        else:
            return out

class dailted_eca_r2_With_cbam(nn.Module):
    def __init__(self, channel, strid,k_size=3, rates=[2, 3, 4], cat=False, res=False):

        super(dailted_eca_r2_With_cbam, self).__init__()

        self.res=res
        self.SG=SpatialGate()
        self.eca_layer1=parallel_dailted_eca_layer(channel, strid,rates=rates,cat=cat)
        self.eca_layer2 = parallel_dailted_eca_layer(channel, strid, rates=rates, cat=cat)
    def forward(self,x) :
        sc=self.eca_layer1(x)
        sc=self.eca_layer2(sc)

        out=self.SG(sc)
        if self.res:
            return x+out
        else:
            return out

class dailted_eca_With_dailted_cbam(nn.Module):
    def __init__(self, channel, strid,k_size=3, rates=[2, 3, 4], cat=False, res=False):
        super(dailted_eca_With_dailted_cbam, self).__init__()

        self.res=res
        self.SG=dailted_SpatialGate(strid,rates)
        self.eca_layer=parallel_dailted_eca_layer(channel, strid,rates=rates,cat=cat)
    def forward(self,x) :
        sc=self.eca_layer(x)
        out=self.SG(sc)
        if self.res:
            return x+out
        else:
            return out

class dailted_eca_Plus_dailted_cbam(nn.Module):
    def __init__(self, channel, strid,k_size=3, rates=[2, 3, 4], cat=False, res=False):
        super(dailted_eca_Plus_dailted_cbam, self).__init__()

        self.res=res
        self.SG=dailted_SpatialGate(strid,rates)
        self.eca_layer=parallel_dailted_eca_layer(channel, strid,rates=rates,cat=cat)
    def forward(self,x) :
        sc=self.eca_layer(x)
        sp=self.SG(x)
        out=sc+sp
        out=F.relu(out, inplace=True)
        if self.res:
            return x+out
        else:
            return out

class dailted_eca_With_dailted_cbam_r2(nn.Module):
    def __init__(self, channel, strid,k_size=3, rates=[2, 3, 4], cat=False, res=False):
        super(dailted_eca_With_dailted_cbam_r2, self).__init__()

        self.res=res
        self.SG1=dailted_SpatialGate(strid,rates)
        self.SG2 = dailted_SpatialGate(strid, rates)
        self.eca_layer=parallel_dailted_eca_layer(channel, strid,rates=rates,cat=cat)
    def forward(self,x) :
        sc=self.eca_layer(x)
        out=self.SG1(sc)
        out=self.SG2(out)
        if self.res:
            return x+out
        else:
            return out

class dailted_SpatialGate(nn.Module):
    def __init__(self,strid,rates,kernel_size=3,all_stage=16):
        super(dailted_SpatialGate, self).__init__()
        self.compress = ChannelPool()

        self.spatials=[]

        self.spatials.append(
             BasicConv(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
                   )
        #other rate
        for r in rates:
            self.spatials.append(
            BasicConv(2, 1, kernel_size=kernel_size, stride=1, dilation=r,padding=r, relu=False)
                            )

        self.spatials = nn.ModuleList(self.spatials)
    def forward(self, x):
        x_compress = self.compress(x)
        out=None
        for f in self.spatials:
            if out == None:
                out = f(x_compress)
            else:
                out = out + f(x_compress)
        scale = torch.sigmoid(out) # broadcasting
        return x * scale



class series_dailted_eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,res=False):
        super(series_dailted_eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # strid=1
        self.res=res
        rates = [strid * r for r in rates]
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.features=[]
        self.features.append(
             nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
                   )
        #other rate
        for r in rates:
            self.features.append(
                nn.Conv1d(1, 1, kernel_size=k_size, dilation=r,padding=r, bias=False)
                            )

        self.features = nn.ModuleList(self.features)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y=y.squeeze(-1).transpose(-1, -2)
        out=None

        for f in self.features:
            if out==None:
                out=f(y)
            else:
                out=f(out)
        out=out.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(out)

        if self.res:
            y = self.relu(x+x * y.expand_as(x))
            return y
        else:

            return x * y.expand_as(x)

class _dailted_eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, rate,k_size=3):
        super(_dailted_eca_layer, self).__init__()
        self.r=rate
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, dilation=self.r,padding=self.r, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Mul_parallel_dailted_eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,res=False):
        super(Mul_parallel_dailted_eca_layer, self).__init__()
        self.cat=cat
        self.res=res
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # strid=1
        rates = [strid * r for r in rates]
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


        self.features=[]
        self.features.append(
            _dailted_eca_layer( channel, rate=1,k_size=3)
                   )
        #other rate
        for r in rates:
            self.features.append(
                _dailted_eca_layer( channel, rate=r,k_size=3))

        self.features = nn.ModuleList(self.features)

        self.bn=nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        out=None
        for f in self.features:
            if out==None:
                out=f(x)
            else:
                out=out+f(x)
        out=self.bn(out)


        return self.relu(out)

class Mul_parallel_dailted_eca_with_cbam(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, strid,k_size=3,rates=[2, 3, 4],cat=False,res=False):
        super(Mul_parallel_dailted_eca_with_cbam, self).__init__()
        self.cat=cat
        self.res = res
        self.SG = SpatialGate()
        self.eca_layer = Mul_parallel_dailted_eca_layer(channel, strid, k_size=k_size, rates=rates, cat=cat)

    def forward(self, x):
        sc = self.eca_layer(x)
        out = self.SG(sc)
        if self.res:
            return x + out
        else:
            return out