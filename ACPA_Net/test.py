import torch
from torch.utils.data import DataLoader
from LossFunctions import SoftDiceLoss
from eval.eval import eval_new, eval_new_for_pr, pre_test, eval_predicted
from utilsS.dataset import BasicDataset, BasicDataset_for_predict, BasicDataset_for_lable
from PIL import Image
import cv2
import numpy as np

def edge_det():
    img = Image.open(
        r'D:\ACPA_Net\data\train\images\AU_images\21.png')

    im_L = img.convert('L')
    # img = np.array(img)

    im_L = np.array(im_L)
    img1 = cv2.Canny(im_L, 10, 70)

    def nothing(x):
        pass
    cv2.createTrackbar('threshold1', 'Canny', 20, 200, nothing)
    cv2.createTrackbar('threshold2', 'Canny', 40, 255, nothing)
    while (1):
        threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
        threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
        img_edges = cv2.Canny(img1, threshold1, threshold2)
        cv2.imshow('original', img1)
        cv2.imshow('Canny', img_edges)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_imgs_dir=r"D:\ACPA_Net\data\CRACK500\val\image"
    val_masks_dir=r"D:\ACPA_Net\data\CRACK500\val\mask"

    net = torch.load(r'D:\ACPA_Net\checkpoints\CP_BestDice_epoch_Allnet.pth')
    net.to(device)
    val_dataset = BasicDataset_for_predict(val_imgs_dir, val_masks_dir, 0.5)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    criterion=SoftDiceLoss()
    val_score=pre_test(net=net, loader=val_loader, device=device, save_path=r'D:/ACPA_Net/pre/')

    print(val_score)