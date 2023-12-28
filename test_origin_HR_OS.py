import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset_HR_OS import BasicDataset, MapDataset

from utils.config_HR_OS import NUM_CLASSES, DATA_PATH, NUM_CHANNELS, GPUS, WEIGHT_PATH
import evaluate_orig_HR_OS

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    if NUM_CHANNELS == 4:
        mean_value = [0.485, 0.456, 0.406, 0.400]
        std_value = [0.229, 0.224, 0.225, 0.225]
    else:
        mean_value = [0.485, 0.456, 0.406]
        std_value = [0.229, 0.224, 0.225]

    # img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_value,
                                std=std_value),
            ])

    img = transform(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            pred = F.softmax(output, dim=1)
            # print("num of class: ", net.n_classes)
        else:
            pred = torch.sigmoid(output)

        pred_max = pred.max(1)[1].cpu().numpy()[0]
    
    return pred_max


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_folder", default=WEIGHT_PATH, type=str, help="weight saved folder for testing")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)
    parser.add_argument("--input", type=str, default=None, required=True,
                        help="path to a single image or image directory")

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument("--gpu_id", type=str, default=GPUS, help="GPU ID")

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    print("python test_origin_HR_OS.py --input data_HR/OS/test/ | tee evaluation_HR_OS.txt")
    args = get_args()
    # in_files = args.input
    input_files = os.path.join(args.input, "imgs")
    out_files = get_output_filenames(args)

    model_file = os.path.join('./best_weights', args.test_folder, 'best.pth')
    # model_file = os.path.join(args.test_folder, 'best.pth')

    net = UNet(n_channels=NUM_CHANNELS, n_classes=NUM_CLASSES)

    logging.info("Loading model {}".format(model_file))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print("gpu_id = ", args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_file, map_location=device))

    logging.info("Model loaded !")

    in_files = []
    if os.path.isdir(input_files):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG', 'tif', 'TIF', 'tiff', 'TIFF']:
            files = glob(os.path.join(input_files, '*.'+ext), recursive=True)
            if len(files)>0:
                in_files.extend(files)
    elif os.path.isfile(input_files):
        in_files.append(input_files)

    result_save_path = os.path.join(args.input, "result", args.test_folder)
    os.makedirs(result_save_path, exist_ok=True)

    for i, fn in enumerate(in_files):
        print("Predicting image {} ...".format(fn))
        img_name = os.path.basename(fn).split('.')[0]

        # img = Image.open(fn)
        if NUM_CHANNELS == 3:
            img = Image.open(fn).convert('RGB')
        elif NUM_CHANNELS == 4:
            img = Image.open(fn).convert('RGBA')
        else:
            print("NUM_CHANNELS is wrong ", NUM_CHANNELS)
        

        pred_max = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            colorized_preds = MapDataset.decode_target(pred_max).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds.save(os.path.join(result_save_path, img_name+'.tif'))
            # for pred in pred_max:
            #     print(pred)

            visible_file = os.path.join(result_save_path, img_name+'_visible.jpg')
            evaluate_orig_HR_OS.save_visible(visible_file, pred_max)

            # logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, pred_max)

    test_label_path = os.path.join(args.input, "masks")
    evaluate_orig_HR_OS.calc_result_excel(result_save_path, test_label_path)