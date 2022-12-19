from utilsS.dataset import BasicDataset

train_imgs_dir=r"D:\ACPA_Net\data\water_leakage\qietu\train\img"
train_masks_dir=r"D:\ACPA_Net\data\water_leakage\qietu\train\annotation"
AU_train_imgs_dir=r'D:\ACPA_Net\data\water_leakage\qietu\train\AU\img'
AU_train_masks_dir=r'D:\ACPA_Net\data\water_leakage\qietu\train\AU\ann'
val_imgs_dir=r"D:\ACPA_Net\data\water_leakage\qietu\val\img"
val_masks_dir=r"D:\ACPA_Net\data\water_leakage\qietu\val\annotation"

CRACK500_tain=r'D:\ACPA_Net\data\CRACK500\train\image'
CRACK500_tain_mask=r'D:\ACPA_Net\data\CRACK500\train\mask'
CRACK500_val=r'D:\ACPA_Net\data\CRACK500\val\image'
CRACK500_val_mask=r'D:\ACPA_Net\data\CRACK500\val\mask'

def crack500_dataset(img_scale,AU):
    train_dataset = BasicDataset(CRACK500_tain, CRACK500_tain_mask, img_scale)
    val_dataset = BasicDataset(CRACK500_val,CRACK500_val_mask, img_scale)
    return train_dataset, val_dataset

def water_dataset(img_scale,AU):
    if AU:
        train_dataset = BasicDataset(AU_train_imgs_dir, AU_train_masks_dir, img_scale)
        val_dataset = BasicDataset(val_imgs_dir, val_masks_dir, img_scale)
    else:
        train_dataset = BasicDataset(train_imgs_dir, train_masks_dir, img_scale)
        val_dataset = BasicDataset(val_imgs_dir, val_masks_dir, img_scale)
    return train_dataset,val_dataset


def get_dataset(dataset,img_scale,AU,wave):

    if dataset=='water':
        train_dataset,val_dataset=water_dataset(img_scale,AU)
        print("success get water dataset!!!")
    elif dataset=='crack500':
        train_dataset, val_dataset =crack500_dataset(img_scale, AU)
        print("success get crack500 dataset!!!")
    return train_dataset,val_dataset

if __name__ == '__main__':
    train,val=get_dataset('Deepcrack',1,AU=True,wave=False)
    print(train.__len__(),val.__len__())
    print(train.__getitem__(0)['image'].shape)
    print(train.__getitem__(0)['mask'].shape)
    print('------------------------------------')