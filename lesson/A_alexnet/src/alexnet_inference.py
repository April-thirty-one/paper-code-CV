# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-13
# @brief      : inference demo
"""
"""
代码功能：
    用AlexNet模型预测一张图片

代码结构：
    1、加载图片
    2、加载模型
    3、模型推理
    4、获取类别   topk --> index -->names
    5、分类结构可视化

注意事项：
    1、模型接收 4D 张量   如：(256, 256, 3) --增加batch_size--> (batch_size, 3, 256, 256)
    2、放弃使用 LRN
    3、增加 AdaptiveAvePool2d   ..自适应池化层 -> 防止输入图片的大小不满足模型要求导致错误
    4、卷积核的数量和论文中的不同
                    conv1   conv2   conv3   conv4   conv5
        paper         96     256     384     384     256
        PyTorch       64     192     384     256     256
"""
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import time
import json
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames: 分类的名字
    :param p_clsnames_cn: 分类的中文名字
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)  ## json.load(文件名) -- 用json.load()函数读取文件句柄，可以直接读取到这个文件中的所有内容，并且读取的结果返回为python的dict对象。
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()  ## file.readlines(void) -- 方法用于读取所有行(直到结束符 EOF)并返回列表
    return class_names, class_names_cn


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()  ## 采用pytorch官方提供的 AlexNet模型
    pretrained_state_dict = torch.load(path_state_dict)  ## 加载 PyTorch 在 ImageNet 上训练得到的参数
    model.load_state_dict(pretrained_state_dict)  ## 加载到自己的模型中
    model.eval()  ## 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def process_img(path_img):
    """
    通过路径加载出图片
    """
    # hard code
    norm_mean = [0.485, 0.456, 0.406]  ## 基于ImageNet统计出的 mean；为什么是小数？ -因为ToTensor()将[0. 255] --> [0, 1]
    norm_std = [0.229, 0.224, 0.225]  ## 基于ImageNet统计出的 std
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),  ## [0, 255] → [0, 1]
        transforms.Normalize(norm_mean, norm_std),  ## 数据归一化处理
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)  ## 78行 定义的 transform
    img_tensor.unsqueeze_(0)  # chw --> bchw  增加一个维度，键knowledge_supplements
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


if __name__ == "__main__":

    # config
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    path_img = os.path.join(BASE_DIR, "..", "data", "Golden Retriever from baidu.jpg")
    # path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")
    path_classnames = os.path.join(BASE_DIR, "..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "imagenet_classnames.txt")

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    alexnet_model = get_model(path_state_dict=path_state_dict, vis_model=True)

    # 3/5 inference  tensor --> vector
    with torch.no_grad():  ## 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。 -- 我们也不需要计算梯度
        time_tic = time.time()
        outputs = alexnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)  ## torch.max() -- 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())  ## .cpu() -- 将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15 + idx * 30, "top {}:{}".format(idx + 1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()

