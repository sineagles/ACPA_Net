import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from eval.PR__curve import PR_Curve
from eval.wirte_excel import write_PR

from utilsS.dataset import BasicDataset
from utilsS.prf_metrics import DIRPF
from utilsS.new_metrics import *
from eval.AverageMeter import AverageMeter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += DIRPF(pred, true_masks)
            pbar.update()
    net.train()
    tot = tot / n_val
    tot[5] = (2 * tot[2] * tot[3]) / (tot[2] + tot[3])
    return tot


def eval_new(net, loader, device, criterion=False, wave=False):
    net.eval()
    avg_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "f1_score": AverageMeter(),
        "val_loss": AverageMeter(),
        "Accuracy": AverageMeter(),
        "specificity": AverageMeter(),
        "sensitivity": AverageMeter(),
        "Auc": AverageMeter(),
        "Ap": AverageMeter(),
    }

    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            x_size = imgs.size()
            W = x_size[-2]
            H = x_size[-1]
            all_npixl = W * H

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if wave:
                Wave_imgs = batch['wave']
                for i, w in enumerate(Wave_imgs):
                    Wave_imgs[i] = Wave_imgs[i].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if wave:
                    mask_pred = net(imgs, Wave_imgs)
                else:
                    mask_pred = net(imgs)

                if criterion:
                    val_loss = criterion(mask_pred, true_masks)
                else:
                    pass

                pred = torch.sigmoid(mask_pred)
                auc = roc_auc_score(true_masks.view(-1).data.cpu().numpy(), pred.view(-1).data.cpu().numpy())
                ap = average_precision_score(true_masks.view(-1).data.cpu().numpy(),
                                             pred.view(-1).data.cpu().numpy())
                pred = (pred > 0.5).float()

                Dice = dice(pred, true_masks)
                Iou = iou(pred, true_masks)
                Recall = recall(pred, true_masks)
                Precision = precision(pred, true_masks)
                F1_Score = fscore(pred, true_masks)

                acc = accuracy(pred, true_masks)
                sp = specificity(pred, true_masks)
                se = sensitivity(pred, true_masks)

                avg_meters['dice'].update(Dice, imgs.size(0))
                avg_meters['iou'].update(Iou, imgs.size(0))
                avg_meters['recall'].update(Recall, imgs.size(0))
                avg_meters['precision'].update(Precision, imgs.size(0))
                avg_meters['f1_score'].update(F1_Score, imgs.size(0))
                avg_meters['val_loss'].update(val_loss.item(), imgs.size(0))
                avg_meters['Accuracy'].update(acc, imgs.size(0))
                avg_meters['specificity'].update(sp, imgs.size(0))
                avg_meters['sensitivity'].update(se, imgs.size(0))
                avg_meters['Auc'].update(auc, imgs.size(0))
                avg_meters['Ap'].update(ap, imgs.size(0))

        pbar.update()
    net.train()

    return avg_meters

def eval_new_for_pr(net, loader, device, criterion=False, wave=False):
    net.eval()
    avg_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "f1_score": AverageMeter(),
        "val_loss": AverageMeter(),
        "Accuracy": AverageMeter(),
        "specificity": AverageMeter(),
        "sensitivity": AverageMeter(),
        "Auc": AverageMeter(),
        "Ap": AverageMeter(),
        "PR_precision": AverageMeter(),
        "PR_recall": AverageMeter(),
        "PR_t": AverageMeter(),
        "PR_F1": AverageMeter(),
    }

    mask_type = torch.float32
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, names = batch['image'], batch['mask'], batch['name']
            x_size = imgs.size()
            W = x_size[-2]
            H = x_size[-1]
            all_npixl = W * H

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if wave:
                Wave_imgs = batch['wave']
                for i, w in enumerate(Wave_imgs):
                    Wave_imgs[i] = Wave_imgs[i].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if wave:
                    mask_pred = net(imgs, Wave_imgs)
                else:
                    mask_pred = net(imgs)

                if criterion:
                    val_loss = criterion(mask_pred, true_masks)
                else:
                    pass

                pred = torch.sigmoid(mask_pred)
                auc = roc_auc_score(true_masks.view(-1).data.cpu().numpy(), pred.view(-1).data.cpu().numpy())  # aucs
                ap = average_precision_score(true_masks.view(-1).data.cpu().numpy(),
                                             pred.view(-1).data.cpu().numpy())

                thresh_list, presicion_list, recall_list, f1_list = PR_Curve(pred, true_masks)
                pred = (pred > 0.5).float()

                Dice = dice(pred, true_masks)
                Iou = iou(pred, true_masks)
                Recall = recall(pred, true_masks)
                Precision = precision(pred, true_masks)
                F1_Score = fscore(pred, true_masks)

                acc = accuracy(pred, true_masks)
                sp = specificity(pred, true_masks)
                se = sensitivity(pred, true_masks)

                avg_meters['dice'].update(Dice, imgs.size(0))
                avg_meters['iou'].update(Iou, imgs.size(0))
                avg_meters['recall'].update(Recall, imgs.size(0))
                avg_meters['precision'].update(Precision, imgs.size(0))
                avg_meters['f1_score'].update(F1_Score, imgs.size(0))
                avg_meters['val_loss'].update(val_loss.item(), imgs.size(0))
                avg_meters['Accuracy'].update(acc, imgs.size(0))
                avg_meters['specificity'].update(sp, imgs.size(0))
                avg_meters['sensitivity'].update(se, imgs.size(0))
                avg_meters['Auc'].update(auc, imgs.size(0))
                avg_meters['Ap'].update(ap, imgs.size(0))
                avg_meters['PR_precision'].update(np.array(presicion_list), imgs.size(0))
                avg_meters['PR_recall'].update(np.array(recall_list), imgs.size(0))
                avg_meters['PR_t'].update(np.array(thresh_list), imgs.size(0))
                avg_meters['PR_F1'].update(np.array(f1_list), imgs.size(0))

            pred = pred.squeeze().cpu().numpy()
            i = 0
            for n in names:
                result = Image.fromarray((pred[i] * 255).astype(np.uint8))
                i = i + 1
                result.save(r'D:/ACPA_Net/pre/' + n)

        pbar.update()
    net.train()

    write_PR(avg_meters['PR_precision'].avg, avg_meters['PR_recall'].avg, avg_meters['PR_t'].avg,
             avg_meters['PR_F1'].avg, "pr.xlsx")

    return avg_meters

def eval_predicted(loader, device):
    avg_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "f1_score": AverageMeter(),
        "val_loss": AverageMeter(),
        "Accuracy": AverageMeter(),
        "specificity": AverageMeter(),
        "sensitivity": AverageMeter(),
        "Ap": AverageMeter(),
    }

    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            pre_label, true_masks = batch['pre_label'], batch['mask']
            x_size = pre_label.size()
            W = x_size[-2]
            H = x_size[-1]
            all_npixl = W * H

            pre_label = pre_label.to(device=device, dtype=mask_type)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            pred = pre_label
            Dice = dice(pred, true_masks)
            Iou = iou(pred, true_masks)
            Recall = recall(pred, true_masks)
            Precision = precision(pred, true_masks)
            F1_Score = fscore(pred, true_masks)

            acc = accuracy(pred, true_masks)
            sp = specificity(pred, true_masks)
            se = sensitivity(pred, true_masks)

            avg_meters['dice'].update(Dice, true_masks.size(0))
            avg_meters['iou'].update(Iou, true_masks.size(0))
            avg_meters['recall'].update(Recall, true_masks.size(0))
            avg_meters['precision'].update(Precision, true_masks.size(0))
            avg_meters['f1_score'].update(F1_Score, true_masks.size(0))
            avg_meters['Accuracy'].update(acc, true_masks.size(0))
            avg_meters['specificity'].update(sp, true_masks.size(0))
            avg_meters['sensitivity'].update(se, true_masks.size(0))
        pbar.update()

    return avg_meters

def pre_test(net, loader, device, save_path=None):
    net.eval()
    avg_meters = {
        "dice": AverageMeter(),
        "PR_precision": AverageMeter(),
        "PR_recall": AverageMeter(),
        "PR_t": AverageMeter(),
        "PR_F1": AverageMeter(),
    }
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, names = batch['image'], batch['mask'], batch['name']
            x_size = imgs.size()
            W = x_size[-2]
            H = x_size[-1]
            all_npixl = W * H

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                pred = torch.sigmoid(mask_pred)
                thresh_list, presicion_list, recall_list, f1_list = PR_Curve(pred, true_masks)
                pred = (pred > 0.5).float()
                Dice = dice(pred, true_masks)
                avg_meters['dice'].update(Dice, imgs.size(0))
                avg_meters['PR_precision'].update(np.array(presicion_list), imgs.size(0))
                avg_meters['PR_recall'].update(np.array(recall_list), imgs.size(0))
                avg_meters['PR_t'].update(np.array(thresh_list), imgs.size(0))
                avg_meters['PR_F1'].update(np.array(f1_list), imgs.size(0))

                pred = pred.squeeze().cpu().numpy()
                i = 0
                for n in names:
                    result = Image.fromarray((pred[i] * 255).astype(np.uint8))
                    i = i + 1
                    result.save(save_path + n)
        pbar.update()

    write_PR(avg_meters['PR_precision'].avg, avg_meters['PR_recall'].avg, avg_meters['PR_t'].avg,
             avg_meters['PR_F1'].avg, "pr.xlsx")
    print("save image to ", (save_path + n))
    return avg_meters['dice'].avg

def pre_test(net, loader, device, save_path=None):
    net.eval()
    avg_meters = {
        "dice": AverageMeter(),
        "PR_precision": AverageMeter(),
        "PR_recall": AverageMeter(),
        "PR_t": AverageMeter(),
        "PR_F1": AverageMeter(),
    }
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, names = batch['image'], batch['mask'], batch['name']
            x_size = imgs.size()
            W = x_size[-2]
            H = x_size[-1]
            all_npixl = W * H

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                pred = torch.sigmoid(mask_pred)
                thresh_list, presicion_list, recall_list, f1_list = PR_Curve(pred, true_masks)
                pred = (pred > 0.5).float()
                Dice = dice(pred, true_masks)
                avg_meters['dice'].update(Dice, imgs.size(0))
                avg_meters['PR_precision'].update(np.array(presicion_list), imgs.size(0))
                avg_meters['PR_recall'].update(np.array(recall_list), imgs.size(0))
                avg_meters['PR_t'].update(np.array(thresh_list), imgs.size(0))
                avg_meters['PR_F1'].update(np.array(f1_list), imgs.size(0))

                pred = pred.squeeze().cpu().numpy()
                i = 0
                for n in names:
                    result = Image.fromarray((pred[i] * 255).astype(np.uint8))
                    i = i + 1
                    result.save(save_path + n)
        pbar.update()

    write_PR(avg_meters['PR_precision'].avg, avg_meters['PR_recall'].avg, avg_meters['PR_t'].avg,
             avg_meters['PR_F1'].avg, "pr.xlsx")
    print("save image to ", (save_path + n))
    return avg_meters['dice'].avg

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='prf', help='[prf | sem]')
parser.add_argument('--model_name', type=str, default='deepcrack')
parser.add_argument('--results_dir', type=str, default='D:\ACPA_Net\results')
parser.add_argument('--gt_dir', type=str, default=r'D:\ACPA_Net\data\Crackdata\test\new_annotation_255\images')
parser.add_argument('--pre_dir', type=str, default=r'D:\ACPA_Net\pre')
parser.add_argument('--suffix_gt', type=str, default='label_viz', help='Suffix of ground-truth file name')
parser.add_argument('--suffix_pred', type=str, default='fused', help='Suffix of predicted file name')
parser.add_argument('--output', type=str, default=r'plots/demo/P-R.prf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()

if __name__ == '__main__':
    val_imgs_dir = r"D:\ACPA_Net\data\CRACK500\val\image"
    val_masks_dir = r"D:\ACPA_Net\data\CRACK500\val\mask'"
    net = torch.load(
        r'D:\ACPA_Net\checkpoints\CP_BestDice_epoch_Allnet.pth')
    val_dataset = BasicDataset(val_imgs_dir, val_masks_dir, 0.5)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    val_score = eval_new(net, val_dataset)
    print(val_score)
