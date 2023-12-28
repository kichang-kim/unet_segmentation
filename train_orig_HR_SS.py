import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from eval_HR_SS import eval_net, validate
from unet import UNet

# from torch.utils.tensorboard import SummaryWriter
from utils.dataset_HR_SS import BasicDataset, MapDataset
from torch.utils.data import DataLoader, random_split

from utils.config_HR_SS import BATCH_SIZE, DATA_PATH, NUM_CLASSES, ignore_label, EPOCHS, GPUS, NUM_CHANNELS, CGT_EPOCHS, MAX_EPOCHS, ACC_CUT_TH
from utils import ext_transforms as et

import gc

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
torch.set_printoptions(profile="full")

torch.cuda.empty_cache()
gc.collect()

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
# dir_checkpoint = 'weights/'


def create_logger_path(log_path):
    final_log_file = os.path.join(log_path, "train.log")
    print("final_log_file", final_log_file)
    head = '%(asctime)-15s %(message)s'
    formatter = logging.Formatter(head)
    #logging.basicConfig(filename=str(final_log_file),
    #                    format=head, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(final_log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_plt_fig(x_data, y_data, x_label, y_label, save_path):
    plt.clf()
    plt.plot(x_data, y_data)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    foldername = os.path.basename(os.path.normpath(save_path))
    filename = os.path.join(save_path, f'{foldername}_{y_label}.png')
    plt.savefig(filename)


def train_net(net,
              device,
              args,
              dir_checkpoint):

    epochs=args.epochs
    batch_size=args.batch_size
    lr=args.lr
    img_scale=args.scale

    if NUM_CHANNELS == 4:
        mean_value = [0.485, 0.456, 0.406, 0.400]
        std_value = [0.229, 0.224, 0.225, 0.225]
    else:
        mean_value = [0.485, 0.456, 0.406]
        std_value = [0.229, 0.224, 0.225]

    train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            #et.ExtRandomScale((0.9, 1.1)),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=False),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean_value,
                            std=std_value),
        ])
    if args.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(args.crop_size),
            et.ExtCenterCrop(args.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean_value,
                            std=std_value),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean_value,
                            std=std_value),
        ])
    if CGT_EPOCHS > 0:
        train_cgt_dst = MapDataset(root=args.data_root,
                                    image_set='cgt_train', transform=train_transform)
        train_cgt_loader = DataLoader(
        train_cgt_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)

    train_dst = MapDataset(root=args.data_root,
                                    image_set='train', transform=train_transform)
    val_dst = MapDataset(root=args.data_root,
                                    image_set='val', transform=val_transform)
    
    train_loader = DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=2)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    best_miou = 0

    logger = create_logger_path(dir_checkpoint)
    logger.info(f'''Starting training:
        Data Root:       {args.data_root}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_dst)}
        Validation size: {len(val_dst)}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.module.n_classes > 1 else 'max', patience=2)
    if epochs > 300:
        LR_DECAY = 0.99
    else:
        LR_DECAY = 0.95
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: LR_DECAY ** epoch,
                                last_epoch=-1)
    if net.module.n_classes > 1:
        # weights = [1.0] * net.module.n_classes
        # weights[0] = 1.0
        # print("weights = ", weights)
        # class_weights = torch.FloatTensor(weights).to(device)
        # criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=class_weights)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    else:
        criterion = nn.BCEWithLogitsLoss()

    epoch_list = []
    loss_list = []
    miou_list = []
    acc_list = []

    if EPOCHS > 100:
        eval_intv = 5
    elif EPOCHS > 50:
        eval_intv = 2
    else:
        eval_intv = 1

    # No coarse-grained data used
    
    # for epoch in range(CGT_EPOCHS):
    #     net.train()
    #     epoch_loss = 0
    #     with tqdm(total=len(train_cgt_dst), desc=f'Epoch {epoch + 1}/{CGT_EPOCHS}', unit='img') as pbar:
    #         for (imgs, true_masks) in train_cgt_loader:
    #             # imgs = batch['image']
    #             # true_masks = batch['mask']
    #             assert imgs.shape[1] == net.module.n_channels, \
    #                 f'Network has been defined with {net.module.n_channels} input channels, ' \
    #                 f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
    #                 'the images are loaded correctly.'

    #             imgs = imgs.to(device=device, dtype=torch.float32)
    #             mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    #             true_masks = true_masks.to(device=device, dtype=mask_type)

    #             masks_pred = net(imgs)
    #             loss = criterion(masks_pred, true_masks)
    #             epoch_loss += loss.item()
    #             # writer.add_scalar('Loss/train', loss.item(), global_step)

    #             pbar.set_postfix(**{'loss (batch)': loss.item()})

    #             optimizer.zero_grad()
    #             loss.backward()
    #             nn.utils.clip_grad_value_(net.parameters(), 0.1)
    #             optimizer.step()

    #             pbar.update(imgs.shape[0])
    #             global_step += 1

    #         if epoch % eval_intv == (eval_intv - 1):
    #             # val_score = eval_net(net, val_loader, device)
    #             valid_loss, mean_IoU, IoU_array, accuracy = validate(net, val_loader, device, criterion)
    #             # msg = 'Epoch: {}, Loss: {:.3f}, MeanIoU: {: 4.4f}, Accuracy: {: 4.4f}'.format(
    #             #        epoch+1, valid_loss, mean_IoU, accuracy)
    #             msg = 'Epoch: {}, Loss: {:.3f}, Accuracy: {: 4.4f}'.format(
    #                     epoch+1, valid_loss, accuracy)
    #             logger.info(msg)
    #             epoch_list.append(epoch+1)
    #             loss_list.append(valid_loss)
    #             miou_list.append(mean_IoU)
    #             acc_list.append(accuracy)

    #         else:
    #             mean_IoU = 0

    #     try:
    #         os.mkdir(dir_checkpoint)
    #         logging.info('Created checkpoint directory')
    #     except OSError:
    #         pass
    #     torch.save(net.module.state_dict(),
    #                 dir_checkpoint + f'/latest.pth')
    #     if mean_IoU > best_miou:
    #         best_miou = mean_IoU
    #         torch.save(net.module.state_dict(),
    #                 dir_checkpoint + f'/best.pth')
    #     logging.info(f'Checkpoint {epoch + 1} saved !')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_dst), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (imgs, true_masks) in train_loader:
                # imgs = batch['image']
                # true_masks = batch['mask']
                assert imgs.shape[1] == net.module.n_channels, \
                    f'Network has been defined with {net.module.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                # print("!!!!!!!!!!!!!!true masks!!!!!!!!!!!!!!!")
                # print(true_masks)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
            scheduler.step()

            if epoch % eval_intv == (eval_intv - 1):
                # val_score = eval_net(net, val_loader, device)
                valid_loss, mean_IoU, IoU_array, accuracy = validate(net, val_loader, device, criterion)
                #msg = 'Epoch: {}, Loss: {:.3f}, MeanIoU: {: 4.4f}, Accuracy: {: 4.4f}'.format(
                #        epoch+1, valid_loss, mean_IoU, accuracy)
                msg = 'Epoch: {}, Loss: {:.3f}, Pixel Accuracy: {: 4.4f}'.format(
                        epoch+1, valid_loss, accuracy)
                logger.info(msg)
                epoch_list.append(epoch+1)
                loss_list.append(valid_loss)
                miou_list.append(mean_IoU)
                acc_list.append(accuracy)

            else:
                mean_IoU = 0

        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.module.state_dict(),
                    dir_checkpoint + f'/latest.pth')
        if mean_IoU > best_miou:
            best_miou = mean_IoU
            torch.save(net.module.state_dict(),
                    dir_checkpoint + f'/best.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

        if epoch >= (EPOCHS-1) and accuracy > ACC_CUT_TH:
            print(f"{accuracy} is over {ACC_CUT_TH}. Stop Training.")
            break

    # writer.close()
    save_plt_fig(epoch_list, loss_list, 'epoch', 'loss', dir_checkpoint)
    save_plt_fig(epoch_list, miou_list, 'epoch', 'Mean IoU', dir_checkpoint)
    save_plt_fig(epoch_list, acc_list, 'epoch', 'Accuracy', dir_checkpoint)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=MAX_EPOCHS,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, #0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument("--gpu_id", type=str, default=GPUS, help="GPU ID")
    parser.add_argument("--data_root", type=str, default=DATA_PATH, help="path to Dataset")
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print("gpu_id = ", args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=NUM_CHANNELS, n_classes=NUM_CLASSES, bilinear=True)

    today = time.localtime()
    folder_path = '{:02d}{:02d}{:02d}_{:02d}{:02d}'.format(today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min)
    weight_path_prefix = "./weights"
    weight_save_path = os.path.join(weight_path_prefix, 'HR_SS', folder_path)
    os.makedirs(weight_save_path, exist_ok=True)
    logging.info(f'Network:\n'
                 f'\t{weight_save_path} weight save folder \n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net = nn.DataParallel(net)
    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net, device=device, args=args, dir_checkpoint=weight_save_path)
        #os.system("python3 test.py --test_folder {}".format(folder_path))
        #print("python3 test.py --input ./data/HR/OS_HR/")
        # os.system("python3 test.py --input ./data/HR/OS_HR/")
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
