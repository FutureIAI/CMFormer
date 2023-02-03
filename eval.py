import cv2
import numpy as np
import torch
import os
from model.CMFormer import CMFormer
from get_config import get_config
from dataloader.NYUv2 import NYUv2, scaleNorm, Normalize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from metric import AverageMeter, accuracy, macc, intersectionAndUnion
from tqdm import tqdm

val_file = r"./datasets/nyudv2/test.txt"
weight_dir = r"./save_weight/CMFormer-S.pth"
vis_save_dir = r"./visualization"
visualize = False

# nyu
label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]


def get_img_dir(file_list):
    images = []
    labels = []
    depths = []
    with open(file_list) as f:
        content = f.readlines()
        for x in content:
            img_name, depth_name, label_name = x.strip().split(' ')
            images += [img_name]
            labels += [label_name]
            depths += [depth_name]
    return images


def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze().transpose([1, 2, 0])

    return colored


Config = get_config()
state_dict = torch.load(weight_dir)
model = CMFormer(Config).to('cuda')
model.load_state_dict(state_dict)
model.eval()

val_dataset = NYUv2(img_dir=val_file,
                    transform=transforms.Compose([
                        scaleNorm(480, 640),
                        ToTensor(),
                        Normalize()
                    ]))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

if __name__ == '__main__':
    img_dir = get_img_dir(val_file)
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    with tqdm(total=len(val_loader), desc=f'test', postfix=dict, mininterval=0.3) as pbar:
        with torch.no_grad():
            for index, sample in enumerate(val_loader):

                image = sample['image'].to('cuda')
                depth = sample['depth'].to('cuda')
                label = sample['label'].to('cuda')
                predict = model(image, depth)

                output = torch.max(predict, 1)[1] + 1
                output = output.squeeze(0)
                acc, pix = accuracy(output, label)
                intersection, union = intersectionAndUnion(output, label, Config.classes)
                acc_meter.update(acc, pix)
                a_m, b_m = macc(output, label, Config.classes)
                intersection_meter.update(intersection)
                union_meter.update(union)
                a_meter.update(a_m)
                b_meter.update(b_m)

                if visualize:
                    name = int(img_dir[index].split('/')[-1][:-4])
                    seg_img = color_label(torch.max(predict, 1)[1] + 1)
                    cv2.imwrite(vis_save_dir + os.sep + "{}seg_img.png".format(str(name)), seg_img)

                pbar.update(1)

            iou = intersection_meter.sum / (union_meter.sum + 1e-10)
            for _iou in iou:
                print(_iou)
            mIoU = round(iou.cpu().numpy().mean() * 100, 2)
            mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
            mAcc = round(mAcc.cpu().numpy().mean() * 100, 2)
            acc = round(acc_meter.average().cpu().numpy() * 100, 2)
            print('===> mIoU: ' + str(mIoU) + '; mPA: ' + str(mAcc) + '; Accuracy: ' + str(acc))
