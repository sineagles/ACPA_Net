import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from config import cfg
from LossFunctions import BinaryFocalLoss,FocalLoss,SoftDiceLoss
from utilsS.new_loss import BCEDiceLoss

def get_loss(args):
    if args.loss=='CE':
        criterion = nn.CrossEntropyLoss() .cuda()
    elif args.loss=='BCE':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss=='BFL':
        criterion = BinaryFocalLoss()
    elif args.loss == 'FL':
        criterion = FocalLoss()
    elif args.loss == 'BCE+DL':
        print("Loading BCE+DL...")
        criterion=BCEDiceLoss()
    else:
        criterion = SoftDiceLoss()
    logging.info(f'Load model {args.loss} successful')
    return criterion

#Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                          targets[i].unsqueeze(0))
        return loss

#Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

