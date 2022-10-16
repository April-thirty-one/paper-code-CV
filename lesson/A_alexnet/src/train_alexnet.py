# -*- coding: utf-8 -*-
"""
# @file name  : train_alexnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-14
# @brief      : alexnet traning
"""
"""
代码功能：
    对 猫狗大战(kaggle) 数据集进行分类
代码结构：
    1、构建DataLoader
    2、构建模型
    3、构建损失函数
    4、构建优化器
    5、迭代训练
"""
import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.models as models
from lesson.A_alexnet.tools.my_dataset import CatDogDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


if __name__ == "__main__":

    # config
    data_dir = os.path.join(BASE_DIR, "..", "data", "train")
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    num_classes = 2    ## 因为想要做 猫狗分类，所以分类数为 2 即可

    MAX_EPOCH = 3       # 可自行修改
    BATCH_SIZE = 128    # 可自行修改
    LR = 0.001          # 可自行修改
    log_interval = 1    # 多少个间隔打印一次 loss 和 acc                      可自行修改
    val_interval = 1    # 多少个 epoch 后执行验证集                          可自行修改
    classes = 2
    start_epoch = -1    ## 因为一开始要 start_epoch += 1, 变相的从 0 开始
    lr_decay_step = 1   # learning_rate衰减的步长 -- 一个 epoch 就要进行衰减   可自行修改

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256)),      # (256, 256) 区别：他们的区别在于处理非正方形的图片，如(512, 1024) -resize(256)-> (256, 512), 但是(512, 1024) -resize(256， 256)-> (256, 256)
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),    ## 256 - 224 = 32; 32(长) * 32(宽) = 1024; 所以，一张图片可以创建出1024张相似但不同的图片
        transforms.RandomHorizontalFlip(p=0.5),  ## 进行水平翻转 -> 1024 * 2 = 2048; 所以，一张图片可以创建出2048张相似但不同的图片
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),    ## TenCrop() -- 将一张图片转变为10张相似但不同的图片
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops], dim=0)),  ## crops就是传进来的10张图片
    ])                  ## transforms.Lambda(lambda x:...) -- 用户可以用transforms.Lambda()函数自行定义transform操作，该操作不是由torchvision库所拥有的，其中参数是lambda表示的是自定义函数。
                        ## torch.stack([tensors], dim=0, out=None) -- 在维度上连接（concatenate）若干个张量。(这些张量形状相同）
    # 构建MyDataset实例
    train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    alexnet_model = get_model(path_state_dict, False)

    ## 做 迁移学习
    num_ftrs = alexnet_model.classifier._modules["6"].in_features               ## in_features -- 输入的特征数; 这句代码就是获得最后一层(分类层的输入参数个数)
    alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)   ## 将最后分类层的输出参数改变为 num_classes=2;

    alexnet_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 是否是用不同 learning_rate 进行训练
    flag = 0
    # flag = 1
    if flag:
        fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
            ## map(function, iterable) -- 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
            ## id(object)  -- 获得object的唯一标识符，标识符是一个整数(内存地址)
        base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},  # 0, 若此处传入0，表示冻结前面的基础层
            {'params': alexnet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)  # 选择优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)  # 设置学习率下降策略
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)

# ============================ step 5/5 训练 ============================
    train_curve = list()     ## 记录曲线
    valid_curve = list()

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        alexnet_model.train()
        for i, data in enumerate(train_loader):

            # if i > 1:
            #     break

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            alexnet_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    bs, ncrops, c, h, w = inputs.size()     # [4, 10, 3, 224, 224
                    outputs = alexnet_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
            alexnet_model.train()

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()





