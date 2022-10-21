'''
@Project ：paper-code-CV 
@File    ：googlenet_v1_visualizaton.py
@IDE     ：PyCharm 
@Author  ：愧序三十 https://github.com/April-thirty-one
@Date    ：2022/10/21 16:33 
'''

import os

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    log_dir = os.path.join(BASE_DIR, "..", "results")

    ## ------------------------------- 卷积核的可视化 -------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel", )

    path_state_dict = os.path.join(BASE_DIR, "..", "data", "googlenet-1378be20.pth")
    googlenet_model = models.googlenet()
    pretrained_state_dict = torch.load(path_state_dict)
    googlenet_model.load_state_dict(pretrained_state_dict)

    kernel_num = -1
    visualization_max = 1
    for sub_moduel in googlenet_model.modules():
        if not isinstance(sub_moduel, nn.Conv2d):
            continue

        kernel_num += 1
        if kernel_num > visualization_max:
            break

        kernels = sub_moduel.weight
        out_c, in_c, k_w, k_h = tuple(kernels.shape)

        if k_w == 1:     ## 我们并不想可视化 1 * 1 的卷积核
            kernel_num -= 1
            continue

        print(f" --- kernels shape = {kernels.shape} --- ")

        for o_idx in range(out_c):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=in_c)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=620)

        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    ## ------------------------------- 特征图的可视化 -------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    path_img = os.path.join(BASE_DIR, "..", "data", "Golden Retriever from baidu.jpg")
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(norm_mean, norm_std)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert("RGB")
    img_tensor = img_transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0)

    path_state_dict = os.path.join(BASE_DIR, "..", "data", "googlenet-1378be20.pth")
    googlenet_model = models.googlenet()
    pretrained_state_dict = torch.load(path_state_dict)
    googlenet_model.load_state_dict(pretrained_state_dict)

    for sub_moduel in googlenet_model.modules():
        if not isinstance(sub_moduel, nn.Conv2d):
            continue
        convlayer = sub_moduel
        break

    feature_map = convlayer(img_tensor)

    feature_map.transpose_(0, 1)
    feature_map_grid = vutils.make_grid(feature_map, normalize=True, scale_each=True, nrow=8)

    writer.add_image("feature map in conv1", feature_map_grid, global_step=620)
    writer.close()