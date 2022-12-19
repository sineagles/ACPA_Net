import argparse
import os
from PIL import Image
import cv2
import numpy as np
import codecs
import glob
import random
from skimage import io, transform

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(args):
    gt_list = glob.glob(os.path.join(args.gt_dir, '*'))
    pred_list = glob.glob(os.path.join(args.pre_dir, '*'))
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=110))
    return pred_imgs, gt_imgs

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)





def save_train_config(args,n_train,n_val,patience,best_epochs):
    with codecs.open(args.wrirte_config_path, 'w', encoding='utf-8') as f:
        line = f'net:' + '\t' + f'{args.net}' + '\n'
        f.write(line)
        line=f'epochs:'+'\t'+f'{args.epochs}'+'\n'
        f.write(line)
        line=f'batch_size:'+'\t'+f'{args.batchsize}'+'\n'
        f.write(line)
        line = f'scale:' + '\t' + f'{args.scale}' + '\n'
        f.write(line)
        line = f'Loss:' + '\t' + f'{args.loss}' + '\n'
        f.write(line)
        line = f'Train size:' + '\t' + f'{n_train}' + '\n'
        f.write(line)
        line = f'Val size:' + '\t' + f'{n_val}' + '\n'
        f.write(line)
        line = f'Deep Supervision:' + '\t' + f'{args.DS}' + '\n'
        f.write(line)
        line = f'Early stopping patience:' + '\t' + f'{patience}' + '\n'
        f.write(line)
        line = f'Best epochs:' + '\t' + f'{best_epochs}' + '\n'
        f.write(line)
        line = f'seed:' + '\t' + f'{args.seed}' + '\n'
        f.write(line)
        line = f'learning rate:' + '\t' + f'{args.lr}' + '\n'
        f.write(line)


def save_val_result(val_result_path,totall_result,lr_list):
    with codecs.open(val_result_path, 'w', encoding='utf-8') as f:

        line = f'epoch' + '\t' + f'Dice' + '\t' + f'Iou' + '\t' + f'Recall' + '\t' + f'Precision' + '\t' +f'F1_score' + '\t' + f'val_loss' + '\t' +\
               f'Accuracy' + '\t' +f'specificity' + '\t' + f'sensitivity'+'\t' + f'Auc'+'\t' + f'Ap'+'\t' +f'lr'+'\n'
        f.write(line)
        for i, ll in enumerate(totall_result):

            line = f'%d' % (i + 1) + '\t' + '%.4f' % ll['dice'].avg + '\t' + '%.4f' % ll[
                'iou'].avg + '\t' + '%.4f' % ll['recall'].avg + '\t' \
                   + '%.4f' % ll['precision'].avg + '\t' + '%.4f' % ll['f1_score'].avg + '\t' +\
                   '%.4f' % ll['val_loss'].avg + '\t' + '%.4f' % ll['Accuracy'].avg+ '\t' +\
                   '%.4f' % ll['specificity'].avg+ '\t' + '%.4f' % ll['sensitivity'].avg+ '\t' + \
                   '%.4f' % ll['Auc'].avg+ '\t'+ '%.4f' % ll['Ap'].avg + '\t' +'%.8f' % lr_list[i]+'\n'
            f.write(line)

def save_train_result(train_result_path,totall_result,lr_list):
    with codecs.open(train_result_path, 'w', encoding='utf-8') as f:
        line = f'epoch' + '\t' + f'Dice' + '\t' + f'Iou' + '\t' + f'avg_loss' + '\n'
        f.write(line)
        for i, ll in enumerate(totall_result):
            line = f'%d'%(i+1)+'\t'+'\t'.join(['%.4f'%v for v in ll])+'\t' + '%.4f' %lr_list[i]+'\n'   #/t表示制表符吧，读完一个ll (表示input-list中的一行就）就1换行
            f.write(line)

def get_image(data_dir):
    img_list = glob.glob(data_dir+'\img\*')
    mask_list=glob.glob(data_dir+r'\annotation_255\*')
    print(len(img_list))
    print('mask_list',len(mask_list))
    imgs = []
    names=[]
    masks=[]

    for img_path in img_list:
        names.append(os.path.basename(img_path))
        img=Image.open(img_path)
        imgs.append(img)
    for mask_path in mask_list:
        mask = Image.open(mask_path)
        masks.append(mask)
    # return names,imgs
    return names, imgs,masks

def get_image_for_cascade_water(data_dir):
    print(data_dir+r'\binary_img_all')

    img_list = glob.glob(r'D:\ACPANet\data\water_leakage\cascade_maskrcnn_water\val\color_img_all\*')
    # mask_list=glob.glob(data_dir+r'\color_img_all\*')
    mask_list = glob.glob(r'D:\ACPANet\data\water_leakage\cascade_maskrcnn_water\val\binary_img_all\*')
    print(len(img_list))
    print('mask_list',len(mask_list))
    imgs = []
    names=[]
    masks=[]

    for img_path in img_list:
        names.append(os.path.basename(img_path))
        img=Image.open(img_path)
        imgs.append(img)
    for mask_path in mask_list:
        mask = Image.open(mask_path)
        masks.append(mask)

    return names, imgs,masks


def get_image_forcell(data_dir):
    img_list = glob.glob(data_dir+'\imgs'+'\*')
    print(img_list)
    print('==================')
    mask_list= glob.glob(data_dir+r'\anns'+'\*')
    print(mask_list)
    imgs = []
    masks = []
    names=[]
    for img_path in img_list:
        names.append(os.path.basename(img_path))
        img = io.imread(img_path)[:, :, :3]

        imgs.append(img)

    for mask_path in mask_list:
        mask = io.imread(mask_path)
        masks.append(mask)
    return names,imgs,masks

def make_label(mask,Threshold):
    mask = mask.convert("L")
    mask = np.array(mask).astype(int)


    label_1 = (mask > Threshold) *1
    label_255=label_1*255

    label_1 = label_1.astype(np.uint8)
    label_255 = label_255.astype(np.uint8)

    label_1 = Image.fromarray(label_1)
    label_255= Image.fromarray(label_255)
    return label_1,label_255


def resize_image(args):
        img_names,imgs=get_image(args.imgs_dir)
        mask_names,masks=get_image(args.masks_dir)
        assert len(imgs)==len(masks), 'len(imgs_dir) and len(masks_dir) must be the same'
        for i in range(len(imgs)):
            re_img = imgs[i].resize((args.w, args.h), Image.BICUBIC)
            re_mask= masks[i].resize((args.w, args.h), Image.BICUBIC)
            re_label_1,re_label_255=make_label(re_mask,args.Threshold)
            re_img.save(os.path.join(args.re_img_dir, img_names[i]))
            re_label_1.save(os.path.join(args.re_mask_dir_new, mask_names[i]))
            re_label_255.save(os.path.join(args.re_mask_dir, mask_names[i]))


def split_dataset(args):
    img_names, imgs = get_image(args.re_img_dir)
    # mask_names, masks = get_image(args.re_mask_dir_new)
    mask_names_255, masks_255 = get_image(args.re_mask_dir)
    assert len(masks_255) == len(imgs), 'len(imgs_dir) and len(masks_dir) must be the same'
    indexs=[i for i in range(len(imgs))]
    print(len(imgs))
    random.shuffle(indexs)
    print(indexs)
    print(type(indexs))

    if args.is_split_test:
        cnt_train=int(round(len(imgs)*0.6,0))
        print(cnt_train)
        cnt_val = int(round(len(imgs) * 0.2, 0))
        print(cnt_val)
        cnt_test = int(round(len(imgs) *0.2, 0))

        train_indexs=indexs[0:cnt_train]
        print(train_indexs)
        val_indexs=indexs[cnt_train:(cnt_train+cnt_val):1]
        test_indexs=indexs[(cnt_train+cnt_val):]


    else:
        cnt_train = 580
        cnt_val=90
        train_indexs = indexs[0:cnt_train]
        print(train_indexs)
        val_indexs = indexs[cnt_train:]

    for i in train_indexs:
        imgs[i].save(os.path.join(args.train_dir,'imgs', img_names[i]))
        masks_255[i].save(os.path.join(args.train_dir, 'anns', mask_names_255[i]))
    for i in val_indexs:
        imgs[i].save(os.path.join(args.val_dir, 'imgs', img_names[i]))
        masks_255[i].save(os.path.join(args.val_dir, 'anns', mask_names_255[i]))


def get_args():
    parser = argparse.ArgumentParser(description='Resize the images and the masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imgs_dir', type=str, default=r'D:\ACPANet\data\water_leakage\qietu\img',
                        help='Original img path')
    parser.add_argument('--masks_dir', type=str, default=r'D:\ACPANet\data\water_leakage\qietu\new_annotations_255',
                        help='Original mask path,whose gray value is 255 or 0.')
    parser.add_argument('--re_img_dir', type=str, default=r'D:\ACPANet\data\cell nuclei\new_data\images',
                        help='The resized img path')
    parser.add_argument('--re_mask_dir', type=str, default=r'D:\ACPANet\data\cell nuclei\new_data\anns',
                        help='The resized mask path,whose bigget gray value is 255.')
    parser.add_argument('--re_mask_dir_new', type=str, default=r'D:\ACPANet\data\cell nuclei\new_data\new_anns',
                        help='The mask which can be the ground truth.( The gray value is 0 or 1)')
    parser.add_argument('--Threshold',  type=int,  default=70)
    parser.add_argument('--w', type=int, default=512,help='The new width')
    parser.add_argument('--h', type=int, default=512,help='The new height')
    parser.add_argument('--train_dir', type=str, default=r'D:\ACPANet\data\cell nuclei\new_data\train',
                        help='train img path')
    parser.add_argument('--val_dir', type=str, default=r'D:\ACPANet\data\cell nuclei\new_data\val',
                        help='val img path')
    parser.add_argument('--test_dir', type=str, default=r'D:\ACPANet\data\water_leakage\qietu\test',
                        help='test img path')
    parser.add_argument('--is_split_test', type=str, default=False,
                        help='wether to split test dataset')

    return parser.parse_args()

if __name__ == '__main__':
    args=get_args()
    name,img,mask=get_image_forcell(r'D:\ACPA_Net\data\cell nuclei\new_data\val')
    print(name)
    print(len(img),len(mask))