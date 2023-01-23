import matplotlib
import matplotlib.colors
import numpy as np
from torch.utils.data import Dataset
import skimage.transform
import random
import torchvision
import torch
import cv2


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
    return {'images': images, 'depths': depths, 'labels': labels}


class NYUv2(Dataset):
    def __init__(self, img_dir=None, transform=None):
        super(NYUv2, self).__init__()
        self.data = get_img_dir(img_dir)
        self.train_img = self.data["images"]
        self.train_depth = self.data["depths"]
        self.train_label = self.data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.train_img)

    def __getitem__(self, item):
        if self.train_img[item][-4:] == '.npy':
            image = np.load(self.train_img[item])
            depth = np.load(self.train_depth[item])
            label = np.load(self.train_label[item])
        else:
            image = cv2.imread(self.train_img[item], 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(self.train_depth[item], 0)
            label = cv2.imread(self.train_label[item], 0)

        sample = {'image': image, 'depth': depth, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __init__(self, h, w):
        self.image_h = h
        self.image_w = w

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # # Bi-linear
        image = skimage.transform.resize(image, (self.image_h, self.image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (self.image_h, self.image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (self.image_h, self.image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + self.th, j:j + self.tw, :],
                'depth': depth[i:i + self.th, j:j + self.tw],
                'label': label[i:i + self.th, j:j + self.tw]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(
            image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label / 1.0).float()}
