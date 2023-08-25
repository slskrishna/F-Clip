import os.path as osp
import json
import cv2
from skimage import io
import numpy as np
import random
from torch.utils.data.dataloader import default_collate

import copy
import os
import math
from random import randint as ri

import albumentations as A

transform_1 = A.Compose([
    A.RandomBrightness(p=0.2),
])

transform_2 = A.Compose([
    A.GaussNoise(p=0.8),
])

transform_3 = A.Compose([
    A.RandomContrast(p=0.8),
])

transform_4 = A.Compose([
    A.Sharpen(p=0.8),
])

transform_5 = A.Compose([
    A.RandomSnow(p=0.8),
])

transform_6 = A.Compose([
    A.RandomFog(p=0.8),
])

transform_7 = A.Compose([
    A.RandomRain(p=0.8),
])

transform_8 = A.Compose([
    A.RandomShadow(p=0.8),
])

transform_9 = A.Compose([
    A.InvertImg(p=0.8),
])

transform_10 = A.Compose([
    A.FancyPCA(p=0.8),
])

transform_11 = A.Compose([
    A.RandomShadow(p=0.8),
])

transform_12 = A.Compose([
    A.ColorJitter(p=0.8),
])

transform_13 = A.Compose([
    A.Superpixels(p=0.8),
])

transform_14 = A.Compose([
    A.PixelDropout(p=0.8),
])

transform_15 = A.Compose([
    A.Emboss(p=0.8),
])

transform_16 = A.Compose([
    A.Downscale(p=0.8),
])

transform_17 = A.Compose([
    A.MultiplicativeNoise(p=0.8),
])

transform_18 = A.Compose([
    A.Posterize(p=0.8),
])

transform_19 = A.Compose([
    A.Solarize(p=0.8),
])

transform_20 = A.Compose([
    A.Equalize(p=0.8),
])

transform_21 = A.Compose([
    A.HueSaturationValue(p=0.8),
])

transform_22 = A.Compose([
    A.RGBShift(p=0.8),
])

transform_23 = A.Compose([
    A.ChannelShuffle(p=0.8),
])

transform_24 = A.Compose([
    A.ToGray(p=0.8),
])

transform_25 = A.Compose([
    A.ToSepia(p=0.8),
])


class Augmentations():
    def __init__(self, transform=None):
        self.transform = transform
        occ_folder = "./data/Stock images/"
        self.occ_files = [occ_folder + file for file in os.listdir(occ_folder) if "DS_" not in file]
        self.occ_length = len(self.occ_files)
        self.outer_contour = json.load(open("data/train_outer_contour.json"))

    def apply_augmentations(self, image, filename, ann={}):
        width = ann.get('width', 512)
        height = ann.get('height', 512)

        if random.randint(1, 10) <= 5:
            mask = cv2.imread(self.occ_files[ri(0, self.occ_length - 1)])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            sx, sy, hm, wm = ri(165, 286), ri(165, 286), len(mask), len(mask[0])
            lines = self.outer_contour.get(filename, [])
            ratio = math.sqrt(random.choice([4900, 6400, 8100, 10000, 12100, 14400, 16900]) / (hm * wm))
            hm, wm = int(ratio * hm), int(wm * ratio)
            mask = cv2.resize(mask, (wm, hm))
            if lines:
                idx = random.randint(0, len(lines) - 1)
                [x1, y1], [x2, y2] = lines[idx], lines[(idx + 1) % len(lines)]
                ratio = random.randint(1, 99) / 100
                sx, sy = int(x1 + ratio * (x2 - x1)), int(y1 + ratio * (y2 - y1))
                sx, sy = max(0, sx - wm // 2), max(0, sy - hm // 2)

            mask_of_mask = (mask[:, :, 0] < 160) | (mask[:, :, 1] < 160) | (mask[:, :, 2] < 160)
            try:
                mask = np.clip((mask * 0.4) + 30, 0, 255)
                image[sy: sy + hm, sx: sx + wm][mask_of_mask] = mask[mask_of_mask]
            except:
                pass

        if random.randint(1, 10) <= 4:
            image = cv2.GaussianBlur(image, (2 * random.randint(1, 3) + 1, 2 * random.randint(1, 3) + 1), 0)

        if random.randint(1, 10) <= 6:
            brightness = random.randint(0, 50)
            contrast = random.randint(7, 13) / 10
            image = np.clip((image * contrast) + brightness, 0, 255)

        # reminder = random.randint(0, 15)
        reminder = random.randint(4, 7)
        if reminder == 1:
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]

        elif reminder == 2:
            image = image[::-1, :, :]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]

        elif reminder == 3:
            image = image[::-1, ::-1, :]
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]

        elif reminder == 4:
            transformed_4 = transform_4(image=image)
            image = transformed_4["image"]

        elif reminder == 5:
            transformed_17 = transform_17(image=image)
            image = transformed_17["image"]

        elif reminder == 6:
            transformed_25 = transform_25(image=image)
            image = transformed_25["image"]

        elif reminder == 7:
            transformed_23 = transform_23(image=image)
            image = transformed_23["image"]

        elif reminder >= 8:
            angle = random.randint(1, 3)
            M = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * 90, 1)
            image = cv2.warpAffine(image, M, (height, width))
            ann['junctions'] -= 255.5
            ann['junctions'] = np.dot(ann['junctions'],
                                      cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle * -90, 1))[:,
                               :2]
            ann['junctions'] += 255.5
            ann['junctions'] = ann['junctions'].astype('float32')
        else:
            pass

        return image, ann


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
