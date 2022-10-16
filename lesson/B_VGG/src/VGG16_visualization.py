"""
代码功能：
1. 卷积核的可视化
2. 特征图的可视化
"""

import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import torchvision.utils as vutlis
import torchvision.models as models

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    log_dir = os.path.join(BASE_DIR, "..", "results")

    ## ------------------------- 卷积核可视化 -------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")

    path_state_dict = os.path.join(BASE_DIR, "..", "data", "vgg16-397923af.pth")
    vggnet = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    vggnet.load_state_dict(pretrained_state_dict)

    kernel_num = -1
    MAX_VIS_KERNEL = 0
    for sub_module in vggnet.modules():
        if not isinstance(sub_module, nn.Conv2d):
            continue

        kernel_num += 1
        if kernel_num > MAX_VIS_KERNEL:
            break

        kernels = sub_module.weight
        out_c, in_c, k_h, k_w = kernels.shape
        print(f" ---- kernels.shape = {kernels.shape} --- ")

        for out_idx in range(out_c):
            kernels_idx = kernels[out_idx, :, :, :].unsqueeze(1)
            kernel_grid = vutlis.make_grid(kernels_idx, normalize=True, scale_each=True, nrow=in_c)
            writer.add_image(f"{kernel_num} Convlayer split in channel", kernel_grid, global_step=out_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutlis.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
        writer.add_image(f"{kernel_num}_all", kernel_grid, global_step=620)

        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    ## ------------------------- 特征图可视化 -------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])


    index = [0, 2]


    img_pil = Image.open(path_img).convert("RGB")
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)

    convlayer1 = vggnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    fmap_1.transpose_(0, 1)
    fmap_1_grid = vutlis.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image("feature map in conv1", fmap_1_grid, global_step=620)
    writer.close()
