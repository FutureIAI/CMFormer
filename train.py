import time
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torchvision import transforms
from lr_policy import WarmUpPolyLR
from torch.utils.data import DataLoader
from model.CMFormer import CMFormer
from loss import CrossEntropyLoss2d
from dataloader.NYUv2 import NYUv2, RandomHSV, scaleNorm, RandomScale, RandomCrop, RandomFlip, Normalize, ToTensor
from get_config import get_config
from metric import AverageMeter, accuracy, macc, intersectionAndUnion

Config = get_config()

device = 'cuda'
med_frq = []
weight_path = './nyuv2_40class_weight.txt'
with open(weight_path, 'r', encoding='utf-8') as f:
    context = f.readlines()

for x in context[1:]:
    x = x.strip().strip('\ufeff')
    med_frq.append(float(x))
med_frq = torch.from_numpy(np.array(med_frq)).float()


def load_weight(model, weight_dir):
    pretrain_weight = OrderedDict()
    weight = torch.load(weight_dir)
    # print("预训练网络有{}个权重".format(len(weight.keys())))
    for key1, value1 in model.state_dict().items():
        for key2, value2 in weight.items():
            if key2 in key1 and value2.shape == value1.shape and 'gfa' not in key1 and 'cac' not in key1:
                pretrain_weight[key1] = value2
                break
            if 'depth_patch_embed1.proj.weight' in key1:
                pretrain_weight[key1] = torch.mean(weight["patch_embed1.proj.weight"], 1).data.view_as(
                    model.state_dict()[key1])
    # print("共提取{}个权重".format(len(pretrain_weight.keys())))
    return pretrain_weight


def train():
    train_dataset = NYUv2(img_dir=Config.train_file,
                          transform=transforms.Compose([
                              scaleNorm(Config.image_h, Config.image_w),
                              RandomScale((1.0, 1.4)),
                              RandomHSV((0.9, 1.1),
                                        (0.9, 1.1),
                                        (25, 25)),
                              RandomCrop(Config.image_h, Config.image_w),
                              RandomFlip(),
                              ToTensor(),
                              Normalize()
                          ]))
    val_dataset = NYUv2(img_dir=Config.val_file,
                        transform=transforms.Compose([
                            scaleNorm(Config.image_h, Config.image_w),
                            ToTensor(),
                            Normalize()
                        ]))
    train_loader = DataLoader(train_dataset, batch_size=Config.train_batch_size, shuffle=True,
                              num_workers=Config.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.val_batch_size, shuffle=False,
                            num_workers=Config.num_workers, pin_memory=True)
    model = CMFormer(Config).to(device)

    if Config.pretrain_weight_dir is not None:
        pretrain_weight = load_weight(model, Config.pretrain_weight_dir)
        # print("网络共有{}个权重，有{}个未被加载".format(len(model.state_dict().keys()),
        #                                   len(model.load_state_dict(pretrain_weight, strict=False)[0])))
        print("未加载的权重为：")
        for weight in list(model.load_state_dict(pretrain_weight, strict=False))[0]:
            print(weight)
        model.load_state_dict(pretrain_weight, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.optimizer.lr,
                                  weight_decay=Config.optimizer.wd)
    criterion = CrossEntropyLoss2d(weight=med_frq).to(device)
    lr_policy = WarmUpPolyLR(Config.optimizer.lr, Config.lr_scheduler.power,
                             len(train_loader) * Config.stop_epoch,
                             len(train_loader) * Config.lr_scheduler.warm_up_epoch)
    step_train = len(train_loader)
    step_val = len(val_loader)

    now_time = time.strftime("%Y-%m-%d#%H-%M-%S", time.localtime())
    log_dir = r"./logs/{}.txt".format(now_time)
    with open(log_dir, "a") as f:
        f.write(Config.train_file)
        f.write("\n")
        f.close()

    metric = []
    iou_list = []
    for epoch in range(Config.stop_epoch - Config.begin_epoch):
        model.train()
        time.sleep(1)
        with tqdm(total=step_train, desc=f'Epoch {epoch}/{Config.stop_epoch - Config.begin_epoch}',
                  postfix=dict, mininterval=0.3) as pbar:
            train_loss = []
            for iteration, sample in enumerate(train_loader):
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                predict = model(image, depth)
                loss = criterion(predict, label)
                train_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_idx = epoch * len(train_loader) + iteration
                lr = lr_policy.get_lr(current_idx)
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                pbar.set_postfix(**{'train_loss': format(sum(train_loss) / len(train_loss), '.5f'),
                                    'lr': format(lr, '.8f')})
                pbar.update(1)

        with torch.no_grad():
            with tqdm(total=step_val, desc=f'validation', postfix=dict, mininterval=0.3) as pbar:
                acc_meter = AverageMeter()
                intersection_meter = AverageMeter()
                union_meter = AverageMeter()
                a_meter = AverageMeter()
                b_meter = AverageMeter()
                model.eval()
                for iteration, sample in enumerate(val_loader):
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
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
                    pbar.update(1)

            iou = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = round(iou.cpu().numpy().mean() * 100, 2)
            mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
            mAcc = round(mAcc.cpu().numpy().mean() * 100, 2)
            acc = round(acc_meter.average().cpu().numpy() * 100, 2)

            print('===> mIoU: ' + str(mIoU) + '; mPA: ' + str(mAcc) + '; Accuracy: ' + str(acc))
            for _iou in iou:
                print(_iou)

            metric.append({'epoch': epoch, 'mIoU': mIoU, 'mPA': mAcc, 'Accuracy': acc})
            iou_list.append(mIoU)
            print('best result:', metric[iou_list.index(max(iou_list))])

            with open(log_dir, "a") as f:
                f.write("epoch{}, train_loss={:.6f}, Accuracy={:.2f}, mPA={:.2f}, mIou={:.4}".format(
                    epoch, sum(train_loss) / len(train_loss), acc, mAcc, mIoU))
                f.write("\n")

        if (epoch + 1) % Config.save_freq == 0 and epoch != Config.begin_epoch:
            state_dict = model.state_dict()
            torch.save(state_dict, "./save_weight/epoch{}-mIoU{}-mPA{}-Accuracy{}.pth".format(
                epoch, mIoU, mAcc, acc))
        if iou_list.index(max(iou_list)) == epoch:
            state_dict = model.state_dict()
            torch.save(state_dict, "./save_weight/best_result.pth")


if __name__ == '__main__':
    train()
