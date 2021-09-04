import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import skimage.io
import skimage.transform
import numpy as np

class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name2 = os.listdir(os.path.join(path, "1st_manual"))
        self.name1.sort()
        self.name2.sort()

        ### Uncomment these lines to create a dataset with a single image ###
        #self.name1 = self.name1[0:1]
        #self.name2 = self.name2[0:1]
        ### Comment these lines if uncommenting the lines above ###
        self.name1 = self.name1
        self.name2 = self.name2

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.name1)

    def __trans__(self, img, size):

        h, w = img.shape[0:2]

        _w = _h = size

        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left

        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):

        name1 = self.name1[index]
        name2 = self.name2[index]
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]

        len_y = 592
        len_x = 592
        im = skimage.io.imread(os.path.join(img_path[0], name1))

        label = skimage.io.imread(os.path.join(img_path[1], name2))
        label = label.reshape((label.shape[0],label.shape[1],1))
        im= im.astype(np.float32, copy=False)/255.
        label = label.astype(np.float32, copy=False)/255.

        temp = np.copy(im)
        im = np.zeros((len_y,len_x,3), dtype=temp.dtype)
        im[:temp.shape[0],:temp.shape[1],:] = temp
        temp = np.copy(label)
        label = np.zeros((len_y,len_x,1), dtype=temp.dtype)
        label[:temp.shape[0],:temp.shape[1],:] = temp

        label = label>=0.5 
        img = np.moveaxis(im, -1, 0)
        label = np.moveaxis(label, -1, 0)

        return {'img':img, 'label':label, 'img_name': name1}

