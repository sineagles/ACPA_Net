import math
import cv2
import numpy as np
from torchsummary import summary

from Attention.ECA import eca_layer, parallel_dailted_eca_layer, dailted_eca_With_emau, dailted_eca_With_dailted_cbam, \
    dailted_eca_With_cbam, parallel_dailted_eca_layer_r2, new_dailted_eca_layer, new_dailted_eca_layer2, \
    new_dailted_eca_layer2_cbam, dailted_eca_Plus_dailted_cbam, new_dailted_eca_layer2_dailted_cbam, \
    dailted_eca_r2_With_cbam, series_dailted_eca_layer, Mul_parallel_dailted_eca_layer, \
    Mul_parallel_dailted_eca_with_cbam, cascade_dailted_eca_layer
# from unet.res_Unet.original_pasnet import PSAModule
from Attention.psa import PSAModule
from Attention.scse import scSE
from unet.unet_parts import *
from Attention.PAM_CAM import DANetHead
from Attention.SE import SELayer,SEWeightModule
from Attention.cbam import CBAM, SpatialGate
from Attention.grid_attention_layer import GridAttentionBlock2D
from unet.mynn import Norm2d
from Attention.EMAU import EMAU
import dropblock

class ACPA_Net(nn.Module):
    def  __init__(self, n_channels, n_classes,deep_supervision, bilinear=True,G_Attention=False,Cat_Attention=False,res=False,cat=False,
                 Attentiondown='dailted_eca',Attentionup='dailted_eca',layer_n=5,fusion=False,edge_fuse=False,
                 dropBlock=False,bn=False, drop_prob=0.1, block_size=5,nr_steps=5000):
        super(ACPA_Net, self).__init__()
        rate1 = [2,4,8]
        rate2 = [4, 8, 16]
        rate3=[6,12,24]
        rate4=[8,16,32]

        rates1 = [3]
        rates2 = [3,9]
        rates3 = [3, 9, 27]
        rates4 = [3, 9, 27,81]
        rates=[3, 9, 27]

        self.bn=bn
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.emau_k=16

        self.fuse=fusion
        self.edge_fuse=edge_fuse
        self.Dblock=dropBlock

        self.layer_n=layer_n
        self.attentiondown=Attentiondown
        self.attentionup = Attentionup
        self.DS=deep_supervision

        self.G_Attention=G_Attention
        self.Cat_Attention=Cat_Attention
        self.mode = 'concatenation'

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0,
            stop_value=drop_prob,
            nr_steps=nr_steps
        )

        if self.Dblock:
            print('Using DropBlock！！！')
            self.inc = DoubleConv_DropB(n_channels, filters[0],bn=bn,dropblock=self.dropblock)
            self.down1 = Down_DropB(filters[0], filters[1],bn=bn, dropblock=self.dropblock)
            self.down2 = Down_DropB(filters[1], filters[2],bn=bn, dropblock=self.dropblock)
            self.down3 = Down_DropB(filters[2], filters[3],bn=bn, dropblock=self.dropblock)
            factor = 2 if bilinear else 1
            self.down4 = Down_DropB(filters[3], filters[4] // factor,bn=bn, dropblock=self.dropblock)

            self.up1 = Up_DropB(filters[4], filters[3] // factor,dropblock=self.dropblock,bn=bn )
            self.up2 = Up_DropB(filters[3], filters[2] // factor, dropblock=self.dropblock,bn=bn)
            self.up3 = Up_DropB(filters[2], filters[1] // factor, dropblock=self.dropblock,bn=bn)
            self.up4 = Up_DropB(filters[1], filters[0],dropblock=self.dropblock,bn=bn)

        else:
            self.inc = DoubleConv(n_channels, filters[0])
            self.down0 = Down(filters[0], filters[0])
            self.down1 = Down(filters[0], filters[1])
            self.down2 = Down(filters[1], filters[2])
            self.down3 = Down(filters[2], filters[3])
            factor = 2 if bilinear else 1
            self.down4 = Down(filters[3], filters[4] // factor)

            self.up1 = Up(filters[4], filters[3] // factor, bilinear)
            self.up2 = Up(filters[3], filters[2] // factor, bilinear)
            self.up3 = Up(filters[2], filters[1] // factor, bilinear)
            self.up4 = Up(filters[1], filters[0], bilinear)

        if self.G_Attention:
            self.G_Attention1 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4] // factor, mode=self.mode,
                                       sub_sample_factor=(2, 2))
            self.G_Attention2 = GridAttentionBlock2D(in_channels=filters[2],  gating_channels=filters[3] // factor, mode=self.mode,
                                       sub_sample_factor=(2, 2))
            self.G_Attention3 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2] // factor,mode=self.mode,
                                        sub_sample_factor=(2, 2))
            self.G_Attention4 = GridAttentionBlock2D(in_channels=filters[0],  gating_channels=filters[1] // factor,mode=self.mode,
                                        sub_sample_factor=(2, 2))

        if self.Cat_Attention=='DAHead':
            self.DANetHead1 = DANetHead(1024 // factor, 1024 // factor, Norm2d, Attention='scsa')
            self.DANetHead2 = DANetHead(512, 512, Norm2d, Attention='scsa')
            self.DA1=  DANetHead(128,128, Norm2d, Attention='scsa')
            self.DA2 = DANetHead(256,256,Norm2d,Attention='scsa')
            self.DA3 = DANetHead(512, 512, Norm2d, Attention='scsa')
        elif self.Cat_Attention=='se':
            self.se1 = SELayer(128,reduction=4,res=True)
            self.se2 = SELayer(256,reduction=8,res=True)
            self.se3 = SELayer(512,reduction=16,res=True)

        elif self.Cat_Attention=='emau':
            self.Cat_Attention5= EMAU(filters[-1] // factor,self.emau_k)
            self.Cat_Attention4 = EMAU(filters[-2], self.emau_k)
            self.Cat_Attention3 = EMAU(filters[-3],self.emau_k)
            self.Cat_Attention2 = EMAU(filters[-4],self.emau_k)
            self.Cat_Attention1 = EMAU(filters[0], self.emau_k)

        elif self.Cat_Attention == 'original_ECA':
            self.Cat_Attention1 =eca_layer(channel=filters[0])
            self.Cat_Attention2 =eca_layer(channel=filters[1])
            self.Cat_Attention3 =eca_layer(channel=filters[2])
            self.Cat_Attention4 =eca_layer(channel=filters[3])
            self.Cat_Attention5 = eca_layer(channel=filters[4] // factor)
        elif self.Cat_Attention == 'dailted_eca':
            self.Cat_Attention1 = parallel_dailted_eca_layer(channel=filters[0],strid=1,res=res,cat=cat)
            self.Cat_Attention2 = parallel_dailted_eca_layer(channel=filters[1],strid=2,res=res,cat=cat)
            self.Cat_Attention3 = parallel_dailted_eca_layer(channel=filters[2],strid=4,res=res,cat=cat)
            self.Cat_Attention4 = parallel_dailted_eca_layer(channel=filters[3],strid=8,res=res,cat=cat)
            self.Cat_Attention5 = parallel_dailted_eca_layer(channel=filters[4] // factor ,strid=8,res=res,cat=cat)
        elif self.Cat_Attention == 'dailted_eca_With_cbam':
            self.Cat_Attention1 = dailted_eca_With_cbam(channel=filters[0],strid=1,res=res,cat=cat)
            self.Cat_Attention2 = dailted_eca_With_cbam(channel=filters[1],strid=2,res=res,cat=cat)
            self.Cat_Attention3 = dailted_eca_With_cbam(channel=filters[2],strid=4,res=res,cat=cat)
            self.Cat_Attention4 = dailted_eca_With_cbam(channel=filters[3],strid=8,res=res,cat=cat)
            self.Cat_Attention5 = dailted_eca_With_cbam(channel=filters[4] // factor ,strid=8,res=res,cat=cat)
        elif self.Cat_Attention == "dailted_eca_With_dailted_cbam":
            self.Cat_Attention1 = dailted_eca_With_dailted_cbam(channel=filters[0], strid=1,res=res,cat=cat)
            self.Cat_Attention2 = dailted_eca_With_dailted_cbam(channel=filters[1], strid=2,res=res,cat=cat)
            self.Cat_Attention3 = dailted_eca_With_dailted_cbam(channel=filters[2], strid=4,res=res,cat=cat)
            self.Cat_Attention4 = dailted_eca_With_dailted_cbam(channel=filters[3], strid=8,res=res,cat=cat)
            self.Cat_Attention5 = dailted_eca_With_dailted_cbam(channel=filters[4] // factor, strid=8,res=res,cat=cat)
        else:
            pass

        if self.attentiondown=="cbam" :
            self.Attentiondown1 = CBAM(filters[0], reduction_ratio=8)
            self.Attentiondown2 = CBAM(filters[1], reduction_ratio=16)
            self.Attentiondown3 = CBAM(filters[2], reduction_ratio=16)
            self.Attentiondown4 = CBAM(filters[3], reduction_ratio=16)

        elif self.attentiondown=="se":
            self.Attentiondown1 = SELayer( filters[0],reduction=8, res=False)
            self.Attentiondown2 = SELayer(filters[1],reduction=16, res=False)
            self.Attentiondown3 = SELayer(filters[2] ,reduction=16, res=False)
            self.Attentiondown4 = SELayer(filters[3],reduction=16, res=False)
        elif self.attentiondown=="original_ECA":
            self.Attentiondown1 =eca_layer(channel=filters[0])
            self.Attentiondown2 =eca_layer(channel=filters[1])
            self.Attentiondown3 =eca_layer(channel=filters[2])
            self.Attentiondown4 =eca_layer(channel=filters[3])
        elif self.attentiondown=="dailted_eca":
            self.Attentiondown1 =parallel_dailted_eca_layer(channel=filters[0],strid=1,rates=rate1,res=res,cat=cat)
            self.Attentiondown2 =parallel_dailted_eca_layer(channel=filters[1],strid=2,rates=rate2,res=res,cat=cat)
            self.Attentiondown3 =parallel_dailted_eca_layer(channel=filters[2],strid=4,rates=rate2,res=res,cat=cat)
            self.Attentiondown4 =parallel_dailted_eca_layer(channel=filters[3],strid=8,rates=rate2,res=res,cat=cat)
        elif self.attentiondown=="dailted_eca_With_emau":
            self.Attentiondown1 = dailted_eca_With_emau(channel=filters[0], strid=1,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentiondown2 = dailted_eca_With_emau(channel=filters[1], strid=2,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentiondown3 = dailted_eca_With_emau(channel=filters[2], strid=4,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentiondown4 = dailted_eca_With_emau(channel=filters[3], strid=8,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
        elif self.attentiondown=="dailted_eca_With_cbam":
            self.Attentiondown1 = dailted_eca_With_cbam(channel=filters[0], strid=1,rates=rates,res=res,cat=cat)
            self.Attentiondown2 = dailted_eca_With_cbam(channel=filters[1], strid=2,rates=rates,res=res,cat=cat)
            self.Attentiondown3 = dailted_eca_With_cbam(channel=filters[2], strid=4,rates=rates,res=res,cat=cat)
            self.Attentiondown4 = dailted_eca_With_cbam(channel=filters[3], strid=8,rates=rates,res=res,cat=cat)
        elif self.attentiondown == "dailted_eca_With_dailted_cbam":
            self.Attentiondown1 = dailted_eca_With_dailted_cbam(channel=filters[0], strid=1,rates=rates,res=res,cat=cat)
            self.Attentiondown2 = dailted_eca_With_dailted_cbam(channel=filters[1], strid=2,rates=rates,res=res,cat=cat)
            self.Attentiondown3 = dailted_eca_With_dailted_cbam(channel=filters[2], strid=4,rates=rates,res=res,cat=cat)
            self.Attentiondown4 = dailted_eca_With_dailted_cbam(channel=filters[3], strid=8,rates=rates,res=res,cat=cat)
        elif self.attentiondown == "parallel_dailted_eca_layer_r2":
            self.Attentiondown1 = parallel_dailted_eca_layer_r2(channel=filters[0], strid=1,rates=rates,cat=cat)
            self.Attentiondown2 = parallel_dailted_eca_layer_r2(channel=filters[1], strid=2,rates=rates,cat=cat)
            self.Attentiondown3 = parallel_dailted_eca_layer_r2(channel=filters[2], strid=4,rates=rates,cat=cat)
            self.Attentiondown4 = parallel_dailted_eca_layer_r2(channel=filters[3], strid=8,rates=rates,cat=cat)
        elif self.attentiondown =="dailted_eca_r2_With_cbam":
            self.Attentiondown1 = dailted_eca_r2_With_cbam(channel=filters[0],strid=1,rates=rates,cat=cat)
            self.Attentiondown2 = dailted_eca_r2_With_cbam(channel=filters[1], strid=2,rates=rates,cat=cat)
            self.Attentiondown3 = dailted_eca_r2_With_cbam(channel=filters[2], strid=4,rates=rates,cat=cat)
            self.Attentiondown4 = dailted_eca_r2_With_cbam(channel=filters[3], strid=8,rates=rates,cat=cat)
        elif self.attentiondown == "new_dailted_eca_layer":
            self.Attentiondown1 = new_dailted_eca_layer(channel=filters[0], strid=1,res=res,rates=[2,4,8],cat=cat)
            self.Attentiondown2 = new_dailted_eca_layer(channel=filters[1], strid=2,res=res,rates=rates,cat=cat)
            self.Attentiondown3 = new_dailted_eca_layer(channel=filters[2], strid=4,res=res,rates=rates,cat=cat)
            self.Attentiondown4 = new_dailted_eca_layer(channel=filters[3], strid=8,res=res,rates=rates,cat=cat)
        elif self.attentiondown == "new_dailted_eca_layer2":
            self.Attentiondown1 = new_dailted_eca_layer2(channel=filters[0], strid=1,rates=rates,res=res,cat=cat)
            self.Attentiondown2 = new_dailted_eca_layer2(channel=filters[1], strid=2,rates=rates,res=res,cat=cat)
            self.Attentiondown3 = new_dailted_eca_layer2(channel=filters[2], strid=4,rates=rates,res=res,cat=cat)
            self.Attentiondown4 = new_dailted_eca_layer2(channel=filters[3], strid=8,rates=rates,res=res,cat=cat)
        elif self.attentiondown == "new_dailted_eca_layer2_cbam":
            self.Attentiondown1 = new_dailted_eca_layer2_cbam(channel=filters[0], strid=1,rates=rates,res=res,cat=cat)
            self.Attentiondown2 = new_dailted_eca_layer2_cbam(channel=filters[1], strid=2,rates=rates,res=res,cat=cat)
            self.Attentiondown3 = new_dailted_eca_layer2_cbam(channel=filters[2], strid=4,rates=rates,res=res,cat=cat)
            self.Attentiondown4 = new_dailted_eca_layer2_cbam(channel=filters[3], strid=8,rates=rates,res=res,cat=cat)
        elif self.attentiondown == "dailted_eca_Plus_dailted_cbam":
            self.Attentiondown1 = dailted_eca_Plus_dailted_cbam(channel=filters[0], strid=1,rates=rates, res=res,cat=cat)
            self.Attentiondown2 = dailted_eca_Plus_dailted_cbam(channel=filters[1], strid=2,rates=rates, res=res,cat=cat)
            self.Attentiondown3 = dailted_eca_Plus_dailted_cbam(channel=filters[2], strid=4,rates=rates, res=res,cat=cat)
            self.Attentiondown4 = dailted_eca_Plus_dailted_cbam(channel=filters[3], strid=8,rates=rates,res=res,cat=cat)
        elif self.attentiondown =="new_dailted_eca_layer2_dailted_cbam":
            self.Attentiondown1 = new_dailted_eca_layer2_dailted_cbam(channel=filters[0], strid=1,rates=rates, res=res, cat=cat)
            self.Attentiondown2 = new_dailted_eca_layer2_dailted_cbam(channel=filters[1], strid=2,rates=rates, res=res, cat=cat)
            self.Attentiondown3 = new_dailted_eca_layer2_dailted_cbam(channel=filters[2], strid=4,rates=rates, res=res, cat=cat)
            self.Attentiondown4 = new_dailted_eca_layer2_dailted_cbam(channel=filters[3], strid=8,rates=rates, res=res, cat=cat)
        elif self.attentiondown=="series_dailted_eca_layer":
            self.Attentiondown1 = series_dailted_eca_layer(channel=filters[0], strid=1, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown2 = series_dailted_eca_layer(channel=filters[1], strid=2, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown3 = series_dailted_eca_layer(channel=filters[2], strid=4, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown4 = series_dailted_eca_layer(channel=filters[3], strid=8, rates=rates, res=res,
                                                                      cat=cat)
        elif self.attentiondown == "Mul_parallel_dailted_eca_layer":
            self.Attentiondown1 = Mul_parallel_dailted_eca_layer(channel=filters[0], strid=1, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown2 = Mul_parallel_dailted_eca_layer(channel=filters[1], strid=2, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown3 = Mul_parallel_dailted_eca_layer(channel=filters[2], strid=4, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown4 = Mul_parallel_dailted_eca_layer(channel=filters[3], strid=8, rates=rates, res=res,
                                                                      cat=cat)
        elif self.attentiondown == "Mul_parallel_dailted_eca_with_cbam":
            self.Attentiondown1 = Mul_parallel_dailted_eca_with_cbam(channel=filters[0], strid=1, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown2 = Mul_parallel_dailted_eca_with_cbam(channel=filters[1], strid=2, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown3 = Mul_parallel_dailted_eca_with_cbam(channel=filters[2], strid=4, rates=rates, res=res,
                                                                      cat=cat)
            self.Attentiondown4 = Mul_parallel_dailted_eca_with_cbam(channel=filters[3], strid=8, rates=rates, res=res,
                                                                      cat=cat)
        elif self.attentiondown == "psa+se":

            self.Attentiondown1 = PSAModule(filters[0],filters[0],attention=SEWeightModule,conv_groups=[1,2,4,8])
            self.Attentiondown2 = PSAModule(filters[1],filters[1],attention=SEWeightModule,conv_groups=[1,2,4,8])
            self.Attentiondown3 = PSAModule(filters[2],filters[2],attention=SEWeightModule,conv_groups=[1,4,8,16])
            self.Attentiondown4 = PSAModule(filters[3],filters[3],attention=SEWeightModule,conv_groups=[1,4,8,16])

        elif self.attentiondown == "psa+eca":
            self.Attentiondown1 = PSAModule(filters[0],filters[0],attention=eca_layer,conv_groups=[1,2,4,8])
            self.Attentiondown2 = PSAModule(filters[1],filters[1],attention=eca_layer,conv_groups=[1,2,4,8])
            self.Attentiondown3 = PSAModule(filters[2],filters[2],attention=eca_layer,conv_groups=[1,4,8,16])
            self.Attentiondown4 = PSAModule(filters[3],filters[3],attention=eca_layer,conv_groups=[1,4,8,16])
        elif self.attentiondown == 'SpatialGate':
            self.Attentiondown1 = SpatialGate()
            self.Attentiondown2 = SpatialGate()
            self.Attentiondown3 = SpatialGate()
            self.Attentiondown4 = SpatialGate()
        elif self.attentiondown=='scSE':
            self.Attentiondown1 = scSE(in_channels=filters[0])
            self.Attentiondown2 = scSE(in_channels=filters[1])
            self.Attentiondown3 = scSE(in_channels=filters[2])
            self.Attentiondown4 = scSE(in_channels=filters[3])
        elif self.attentiondown == "cascade_dailted_eca":
            self.Attentiondown1 = cascade_dailted_eca_layer(channel=filters[0], strid=1, rates=rates1, res=res, bn=self.bn)
            self.Attentiondown2 = cascade_dailted_eca_layer(channel=filters[1], strid=2, rates=rates2, res=res, bn=self.bn)
            self.Attentiondown3 = cascade_dailted_eca_layer(channel=filters[2], strid=4, rates=rates3, res=res, bn=self.bn)
            self.Attentiondown4 = cascade_dailted_eca_layer(channel=filters[3], strid=8, rates=rates4, res=res, bn=self.bn)
        else:
            pass

        if self.attentionup=="cbam":
            self.Attentionup1 = CBAM(filters[3] // factor, reduction_ratio=16)
            self.Attentionup2 = CBAM(filters[2] // factor, reduction_ratio=16)
            self.Attentionup3 = CBAM(filters[1] // factor, reduction_ratio=16)
            self.Attentionup4 = CBAM(filters[0], reduction_ratio=8)
        elif self.attentionup=="se":
            self.Attentionup1 = SELayer( filters[3] // factor, reduction=16, res=False)
            self.Attentionup2 = SELayer(filters[2] // factor, reduction=16, res=False)
            self.Attentionup3 = SELayer(filters[1] // factor, reduction=16, res=False)
            self.Attentionup4 = SELayer(filters[0], reduction=8, res=False)
        elif self.attentionup == "original_ECA":
            self.Attentionup1 = eca_layer(channel=filters[3] // factor)
            self.Attentionup2 = eca_layer(filters[2] // factor)
            self.Attentionup3 = eca_layer(filters[1] // factor)
            self.Attentionup4 = eca_layer(channel=filters[0])
        elif self.attentionup == "dailted_eca":
            self.Attentionup1 = parallel_dailted_eca_layer(channel=filters[3] // factor,strid=8,rates=rate2,res=res,cat=cat)
            self.Attentionup2 = parallel_dailted_eca_layer(filters[2] // factor,strid=4,rates=rate2,res=res,cat=cat)
            self.Attentionup3 = parallel_dailted_eca_layer(filters[1] // factor,strid=2,rates=rate2,res=res,cat=cat)
            self.Attentionup4 = parallel_dailted_eca_layer(channel=filters[0],strid=1,rates=rate1,res=res,cat=cat)
        elif self.attentionup == "dailted_eca_With_emau":
            self.Attentionup1 = dailted_eca_With_emau(channel=filters[3] // factor,strid=8,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentionup2 = dailted_eca_With_emau(filters[2] // factor,strid=4,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentionup3 = dailted_eca_With_emau(filters[1] // factor,strid=2,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
            self.Attentionup4 = dailted_eca_With_emau(channel=filters[0],strid=1,emau_k=self.emau_k,rates=rates,res=res,cat=cat)
        elif self.attentionup == "dailted_eca_With_cbam":
            self.Attentionup1 = dailted_eca_With_cbam(channel=filters[3] // factor,strid=8,rates=rates,res=res,cat=cat)
            self.Attentionup2 = dailted_eca_With_cbam(filters[2] // factor,strid=4,rates=rates,res=res,cat=cat)
            self.Attentionup3 = dailted_eca_With_cbam(filters[1] // factor,strid=2,rates=rates,res=res,cat=cat)
            self.Attentionup4 = dailted_eca_With_cbam(channel=filters[0],strid=1,rates=rates,res=res,cat=cat)
        elif self.attentionup == "dailted_eca_With_dailted_cbam":
            self.Attentionup1 = dailted_eca_With_dailted_cbam(channel=filters[3] // factor,strid=8,rates=rates,res=res,cat=cat)
            self.Attentionup2 = dailted_eca_With_dailted_cbam(filters[2] // factor,strid=4,rates=rates,res=res,cat=cat)
            self.Attentionup3 = dailted_eca_With_dailted_cbam(filters[1] // factor,strid=2,rates=rates,res=res,cat=cat)
            self.Attentionup4 = dailted_eca_With_dailted_cbam(channel=filters[0],strid=1,rates=rates,res=res,cat=cat)
        elif self.attentionup == "parallel_dailted_eca_layer_r2":
            self.Attentionup1 = parallel_dailted_eca_layer_r2(channel=filters[3] // factor,strid=8, rates=rates,cat=cat)
            self.Attentionup2 = parallel_dailted_eca_layer_r2(filters[2] // factor, strid=4, rates=rates,cat=cat)
            self.Attentionup3 = parallel_dailted_eca_layer_r2(filters[1] // factor, strid=2, rates=rates,cat=cat)
            self.Attentionup4 = parallel_dailted_eca_layer_r2(channel=filters[0], strid=1, rates=rates,cat=cat)
        elif self.attentionup == "dailted_eca_r2_With_cbam":
            self.Attentionup1 = dailted_eca_r2_With_cbam(channel=filters[3] // factor,strid=8, rates=rates,cat=cat)
            self.Attentionup2 = dailted_eca_r2_With_cbam(filters[2] // factor, strid=4, rates=rates,cat=cat)
            self.Attentionup3 = dailted_eca_r2_With_cbam(filters[1] // factor, strid=2, rates=rates,cat=cat)
            self.Attentionup4 = dailted_eca_r2_With_cbam(channel=filters[0], strid=1, rates=rates,cat=cat)
        elif self.attentionup == "new_dailted_eca_layer":
            self.Attentionup1 = new_dailted_eca_layer(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = new_dailted_eca_layer(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = new_dailted_eca_layer(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = new_dailted_eca_layer(channel=filters[0], strid=1, rates=[2,4,8],res=res,cat=cat)
        elif self.attentionup == "new_dailted_eca_layer2":
            self.Attentionup1 = new_dailted_eca_layer2(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = new_dailted_eca_layer2(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = new_dailted_eca_layer2(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = new_dailted_eca_layer2(channel=filters[0], strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup == "new_dailted_eca_layer2_cbam":
            self.Attentionup1 = new_dailted_eca_layer2_cbam(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = new_dailted_eca_layer2_cbam(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = new_dailted_eca_layer2_cbam(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = new_dailted_eca_layer2_cbam(channel=filters[0], strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup == "dailted_eca_Plus_dailted_cbam":
            self.Attentionup1 = dailted_eca_Plus_dailted_cbam(channel=filters[3] // factor,strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = dailted_eca_Plus_dailted_cbam(filters[2] // factor,strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = dailted_eca_Plus_dailted_cbam(filters[1] // factor,strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = dailted_eca_Plus_dailted_cbam(channel=filters[0],strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup =='new_dailted_eca_layer2_dailted_cbam':
            self.Attentionup1 = new_dailted_eca_layer2_dailted_cbam(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = new_dailted_eca_layer2_dailted_cbam(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = new_dailted_eca_layer2_dailted_cbam(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = new_dailted_eca_layer2_dailted_cbam(channel=filters[0], strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup =='series_dailted_eca_layer':
            self.Attentionup1 = series_dailted_eca_layer(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = series_dailted_eca_layer(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = series_dailted_eca_layer(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = series_dailted_eca_layer(channel=filters[0], strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup =='Mul_parallel_dailted_eca_layer':
            self.Attentionup1 = Mul_parallel_dailted_eca_layer(channel=filters[3] // factor, strid=8, rates=rates,res=res,cat=cat)
            self.Attentionup2 = Mul_parallel_dailted_eca_layer(filters[2] // factor, strid=4, rates=rates,res=res,cat=cat)
            self.Attentionup3 = Mul_parallel_dailted_eca_layer(filters[1] // factor, strid=2, rates=rates,res=res,cat=cat)
            self.Attentionup4 = Mul_parallel_dailted_eca_layer(channel=filters[0], strid=1, rates=rates,res=res,cat=cat)
        elif self.attentionup == 'Mul_parallel_dailted_eca_with_cbam':
            self.Attentionup1 = Mul_parallel_dailted_eca_with_cbam(channel=filters[3] // factor, strid=8, rates=rates,
                                                               res=res, cat=cat)
            self.Attentionup2 = Mul_parallel_dailted_eca_with_cbam(filters[2] // factor, strid=4, rates=rates, res=res,
                                                               cat=cat)
            self.Attentionup3 = Mul_parallel_dailted_eca_with_cbam(filters[1] // factor, strid=2, rates=rates, res=res,
                                                               cat=cat)
            self.Attentionup4 = Mul_parallel_dailted_eca_with_cbam(channel=filters[0], strid=1, rates=rates, res=res,
                                                               cat=cat)
        elif self.attentionup == 'psa+se':
            self.Attentionup1 =  PSAModule(filters[3] // factor,filters[3] // factor,attention=SEWeightModule,conv_groups=[1,4,8,16])
            self.Attentionup2 =  PSAModule(filters[2] // factor,filters[2] // factor,attention=SEWeightModule,conv_groups=[1,4,8,16])
            self.Attentionup3 =  PSAModule(filters[1] // factor,filters[1] // factor,attention=SEWeightModule,conv_groups=[1,2,4,8])
            self.Attentionup4 =  PSAModule(filters[0],filters[0],attention=SEWeightModule,conv_groups=[1,2,4,8])
        elif self.attentionup == 'psa+eca':
            self.Attentionup1 = PSAModule(filters[3] // factor, filters[3] // factor, attention=eca_layer,conv_groups=[1,4,8,16])
            self.Attentionup2 = PSAModule(filters[2] // factor, filters[2] // factor, attention=eca_layer,conv_groups=[1,4,8,16])
            self.Attentionup3 = PSAModule(filters[1] // factor, filters[1] // factor, attention=eca_layer,conv_groups=[1,2,4,8])
            self.Attentionup4 = PSAModule(filters[0], filters[0], attention=eca_layer,conv_groups=[1,2,4,8])
        elif self.attentionup == 'SpatialGate':
            self.Attentionup1 = SpatialGate()
            self.Attentionup2 = SpatialGate()
            self.Attentionup3 = SpatialGate()
            self.Attentionup4 = SpatialGate()
        elif self.attentionup == 'scSE':
            self.Attentionup1 = scSE(filters[3] // factor)
            self.Attentionup2 = scSE(filters[2] // factor)
            self.Attentionup3 = scSE(filters[1] // factor)
            self.Attentionup4 = scSE(filters[0] )
        elif self.attentionup == "cascade_dailted_eca":
            self.Attentionup1 = cascade_dailted_eca_layer(channel=filters[3] // factor,strid=8,rates=rates4,res=res,bn=self.bn)
            self.Attentionup2 = cascade_dailted_eca_layer(filters[2] // factor,strid=4,rates=rates3,res=res,bn=self.bn)
            self.Attentionup3 = cascade_dailted_eca_layer(filters[1] // factor,strid=2,rates=rates2,res=res,bn=self.bn)
            self.Attentionup4 = cascade_dailted_eca_layer(channel=filters[0],strid=1,rates=rates1,res=res,bn=self.bn)
        else:
            pass

        self.side_conv0 = nn.Conv2d(filters[4] // factor,  self.n_classes, kernel_size=1, stride=1, bias=True) #这是直接从编码器输出
        self.side_conv1 = nn.Conv2d(filters[3] // factor,  self.n_classes, kernel_size=1, stride=1, bias=True)
        self.side_conv2 = nn.Conv2d(filters[2] // factor,  self.n_classes, kernel_size=1, stride=1, bias=True)
        self.side_conv3 = nn.Conv2d(filters[1] // factor,  self.n_classes, kernel_size=1, stride=1, bias=True)

        if self.edge_fuse:

            self.outc =nn.Sequential(
                DoubleConv(filters[0]+2,32),
                OutConv(32, n_classes),
            )
        else:
            self.outc = OutConv(filters[0] , n_classes)

        if self.fuse:
            self.side_preFuse_conv0 =  nn.Sequential(
                nn.Conv2d(filters[4] // factor, 32, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.side_preFuse_conv1 =nn.Sequential(
                nn.Conv2d(filters[3]// factor, 32, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.side_preFuse_conv2 =  nn.Sequential(
                nn.Conv2d(filters[2] // factor, 32, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.side_preFuse_conv3 =  nn.Sequential(
                nn.Conv2d(filters[1] // factor, 32, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.side_preFuse_conv4 =   nn.Sequential(
                nn.Conv2d(filters[0], 32, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.fuse_conv= nn.Sequential(
                nn.Conv2d(32*5, 32, kernel_size=3, stride=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.fuse_emau=EMAU (32, self.emau_k)

            self.fuse_out = nn.Conv2d(32, self.n_classes, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("initialization is successful!!!")

    def forward(self, x):
        x_size=x.size()
        W=x_size[-2]
        H=x_size[-1]
        if self.layer_n==5:
            if self.attentiondown:
                x1 = self.inc(x)
                x1=self.Attentiondown1(x1)
                x2 = self.down1(x1)
                x2=self.Attentiondown2(x2)
                x3 = self.down2(x2)
                x3=self.Attentiondown3(x3)
                x4 = self.down3(x3)
                x4=self.Attentiondown4(x4)
                x5=self.down4(x4)
            else:
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)

            if self.Cat_Attention=='emau':
                x5 ,_= self.Cat_Attention5(x5)
                x4 ,_ = self.Cat_Attention4(x4)
                x3 ,_= self.Cat_Attention3(x3)
                x2 ,_= self.Cat_Attention2(x2)
                x1 ,_= self.Cat_Attention1(x1)
            elif self.Cat_Attention==False:
                pass
            else:
                x5 = self.Cat_Attention5(x5)
                x4  = self.Cat_Attention4(x4)
                x3 = self.Cat_Attention3(x3)
                x2 = self.Cat_Attention2(x2)
                x1 = self.Cat_Attention1(x1)

            """Decoder"""
            if self.G_Attention==True :
                if self.attentionup:
                    x4, _ = self.G_Attention1(x4, x5)
                    y1 = self.up1(x5, x4)
                    y1 = self.Attentionup1(y1)
                    x3, _ = self.G_Attention2(x3, y1)
                    y2 = self.up2(y1, x3)
                    y2 = self.Attentionup2(y2)
                    x2, _ = self.G_Attention3(x2, y2)
                    y3 = self.up3(y2, x2)
                    y3 = self.Attentionup3(y3)
                    x1, _ = self.G_Attention4(x1, y3)
                    y4 = self.up4(y3, x1)
                    y4 = self.Attentionup4(y4)

                else:
                    x4,_ = self.G_Attention1(x4,x5)
                    y1 = self.up1(x5,x4)
                    x3,_ =  self.G_Attention2(x3,y1)
                    y2 = self.up2(y1, x3)
                    x2,_ = self.G_Attention3(x2,y2)
                    y3 = self.up3(y2, x2)
                    x1,_ = self.G_Attention4(x1,y3)
                    y4 = self.up4(y3,x1)

            elif self.attentionup:
                y1 = self.up1(x5, x4)
                y1=self.Attentionup1(y1)
                y2 = self.up2(y1, x3)
                y2 = self.Attentionup2(y2)
                y3 = self.up3(y2, x2)
                y3 = self.Attentionup3(y3)
                y4 = self.up4(y3, x1)
                y4 = self.Attentionup4(y4)

            else:
                y1 = self.up1(x5, x4)
                y2 = self.up2(y1, x3)
                y3 = self.up3(y2 , x2)
                y4 = self.up4(y3, x1)

            """Output"""
            outside=[]
            if self.DS=='Decoder':
                # side output features
                side_output0 = self.side_conv0(x5)
                side_output1 = self.side_conv1(y1)
                side_output2 = self.side_conv2(y2)
                side_output3 = self.side_conv3(y3)

                side_output0 = F.interpolate(side_output0, size=(W,H), mode='nearest',
                                             )
                outside.append(side_output0)
                side_output1 = F.interpolate(side_output1, size=(W,H), mode='nearest',
                                             )
                outside.append(side_output1)
                side_output2 = F.interpolate(side_output2, size=(W,H), mode='nearest',
                                             )
                outside.append(side_output2)
                side_output3 = F.interpolate(side_output3, size=(W,H), mode='nearest',
                                             )
                outside.append(side_output3)

            if self.edge_fuse:
                x=x*255
                im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
                canny1 = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
                for i in range(x_size[0]):
                    canny1[i] = cv2.Canny(im_arr[i], 30, 50)
                canny1 = torch.from_numpy(canny1).cuda().float()

                canny2 = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
                for i in range(x_size[0]):
                    canny2[i] = cv2.Canny(im_arr[i], 100, 150)
                canny2 = torch.from_numpy(canny2).cuda().float()

                y4 = torch.cat((y4, canny1,canny2), dim=1)
            else:
                pass

            logits = self.outc(y4)

            outside.append(logits)

            if self.fuse:
                side_preFuse0 = self.side_preFuse_conv0(x5)
                side_preFuse1 = self.side_preFuse_conv1(y1)
                side_preFuse2 = self.side_preFuse_conv2(y2)
                side_preFuse3 = self.side_preFuse_conv3(y3)
                side_preFuse4 = self.side_preFuse_conv4(y4)

                side_preFuse0 = F.interpolate(side_preFuse0, size=(W,H), mode='nearest'
                                              )
                side_preFuse1 = F.interpolate(side_preFuse1, size=(W,H), mode='nearest'
                                              )
                side_preFuse2 = F.interpolate(side_preFuse2, size=(W,H), mode='nearest'
                                              )
                side_preFuse3 = F.interpolate(side_preFuse3, size=(W,H), mode='nearest'
                                              )
                side_preFuse4 = F.interpolate(side_preFuse4, size=(W,H), mode='nearest'
                                              )
                cat_cove = torch.cat((side_preFuse0, side_preFuse1, side_preFuse2, side_preFuse3, side_preFuse4), dim=1)
                fuse = self.fuse_conv(cat_cove)
                fuse,mu=self.fuse_emau(fuse)
                fuse_out=self.fuse_out(fuse)
                fuse_out = F.interpolate(fuse_out, size=(W,H), mode='nearest' )
                outside.append(fuse_out)

        if self.DS=='DAHead':
            assert self.DAhead,f'DAHead should be true'

        if self.training and (not self.DS=='DAHead'):
            return outside
        else:
            return logits

def main():
    model = ACPA_Net(n_channels=3, n_classes=1,deep_supervision='Decoder', bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model,(3,256,256),device='cuda')

if __name__ == '__main__':
    main()

