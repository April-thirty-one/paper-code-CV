# -*- coding: utf-8 -*-
"""
# @file name  : my_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-02-14
# @brief      : 数据集Dataset定义
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        """
        凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
        """
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        """
        __len__()的作用是返回容器中元素的个数，要想使len()函数成功执行，必须要在类中定义__len__()。
        """
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    """ (!! 很重要 !!)
    这个函数可以将信息转化为一个列表，这样配合 __gititem__(self)函数 可以加快查找速度
    """
    def _get_img_info(self):

        img_names = os.listdir(self.data_dir)   ## os.listdir(dir) -- 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))   ## endswith(str) -- 判断字符串是否以指定字符串str结尾，当前用来判断后缀名是否是“jpg”
                         ## filter(function, iterable) -- 用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表, 第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

        random.seed(self.rng_seed)   ## 固定打乱的顺序
        random.shuffle(img_names)

        img_labels = [0 if item.startswith('cat') else 1 for item in img_names]   ## ing_labels -- 猫是0，狗是1

        split_idx = int(len(img_labels) * self.split_n)  # 25000* 0.9 = 22500(前22500是训练数据)
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]     # 数据集90%训练
            # img_set = img_names[:22500]     #  hard code 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        ## 这里 path_img_set 是一个由所有jpg文件的路径组成的列表
        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
