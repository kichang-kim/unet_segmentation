import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from dice_loss import dice_coeff
from utils.config_NR_OS import ignore_label, NUM_CLASSES

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            # imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.module.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks, ignore_index=ignore_label).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.module.train()
    return tot / n_val


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def validate(net, loader, device, criterion): #config, testloader, model):
    net.eval()
    mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
#    n_val = len(loader)
#
#    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
#        for imgs, true_masks in loader:
#            # imgs, true_masks = batch['image'], batch['mask']
#            imgs = imgs.to(device=device, dtype=torch.float32)
#            true_masks = true_masks.to(device=device, dtype=mask_type)
#
#            with torch.no_grad():
#                pred = net(imgs)
#                loss = criterion(pred, true_masks)
#
#                if not isinstance(pred, (list, tuple)):
#                    pred = [pred]

    with torch.no_grad():
        for batch in loader:
            image, label = batch
            size = label.size()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            #if not isinstance(pred, (list, tuple)):
            #    pred = [pred]
            
            #x = F.interpolate(
            #    input=pred, size=size[-2:],
            #    mode='bilinear'
            #)

            confusion_matrix[...] += get_confusion_matrix(
                label,
                pred,
                size,
                NUM_CLASSES,
                ignore_label
            )
            losses = criterion(pred, label)
            loss = losses.mean()
            # if dist.is_distributed():
            #     reduced_loss = reduce_tensor(loss)
            # else:
            reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    # if dist.is_distributed():
    #     confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    #     reduced_confusion_matrix = reduce_tensor(confusion_matrix)
    #     confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    # for i in range(nums):
    #print(confusion_matrix)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))[1:]
    mean_IoU = IoU_array.mean()
    accuracy = tp.sum() / pos.sum()
        # if dist.get_rank() <= 0:
        #     logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    # writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    # writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    # writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array, accuracy
