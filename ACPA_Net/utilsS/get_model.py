from unet.unet_eca import ACPA_Net
import logging

def get_model(args):
    if args.net == 'ACPA_Net':
        net =ACPA_Net(n_channels=3, n_classes=1,deep_supervision=args.DS, bilinear=True)
        logging.info(f'Load model {args.net} successful')
    else:
        print("You got nothing !!!!!!")

    return net