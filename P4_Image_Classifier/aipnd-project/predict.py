# 导入各库
import argparse
import copy
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils

use_gpu = torch.cuda.is_available


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    # 调整图像大小
    image.thumbnail((256, 256))
    # 裁切出中心224*224部分 使用crop方法 这里用image1作临时变量 不做临时变量似乎有问题！？
    image1 = image.crop((16, 16, 240, 240))
    # 归一化
    np_image = np.array(image1) / 255
    # 进行标准化 每个通道减去均值 并除以标准差
    np_image[:, :, 0] = (np_image[:, :, 0] - 0.485) / 0.229
    np_image[:, :, 1] = (np_image[:, :, 1] - 0.456) / 0.224
    np_image[:, :, 2] = (np_image[:, :, 2] - 0.406) / 0.225
    # 更改维度
    pytorch_np_image = np_image.transpose((2, 0, 1))
    return pytorch_np_image


def load_checkpoint(args):
    checkpoint_provided = torch.load(args.saved_model)
    # 使用VGG13 或 VGG16
    if checkpoint_provided['arch'] == 'vgg16':
        model = models.vgg16()
    elif checkpoint_provided['arch'] == 'vgg13':
        model = models.vgg13()
    # 更改分类器
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('0', nn.Linear(num_features, 4096, bias=True)),
        ('1', nn.ReLU(inplace=True)),
        ('2', nn.Dropout(p=0.5)),
        ('3', nn.Linear(4096, 4096, bias=True)),
        ('4', nn.ReLU(inplace=True)),
        ('5', nn.Dropout(p=0.5)),
        ('hidden', nn.Linear(4096, args.hidden_units, bias=True)),
        ('6', nn.Linear(args.hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # 替换分类器
    model.classifier = classifier
    model.load_state_dict(checkpoint_provided['state_dict'])
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print("使用GPU中！")
        else:
            print("使用CPU中！请检查GPU配置")

    class_to_idx = checkpoint_provided['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class


def predict(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # DONE: Implement the code to predict the class from an image file
    model.eval()  # 推理模式
    model.cuda()  # 丢到GPU中
    # 获取并处理图片 并且将图片转为Tensor
    image = torch.FloatTensor(process_image(Image.open(image_path))).cuda()
    # 这里试了半天 参考了 https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list-of-1-values-to-match-the-convolution-dimensions-but-got-stride-1-1/17140
    # 多增加一个维度？？让我有点小懵？输入不是3*224*224嘛 为什么变成1*3*224*224
    a = Variable(image)
    a = a.unsqueeze(0)
    # 前向计算 使用forward报错...似乎是cpu gpu问题 这里就采用cpu计算
    with torch.no_grad():
        output = model.forward(a)
        top_prob, top_labels = torch.topk(output, topk)
        # 做指数运算 复原logsoftmax 这里用一个临时变量 否则不会在原地exp运算
        top_prob_ori = top_prob.exp()
    # 取出索引表来做给列表
    class_to_idx_inv = {class_to_idx[k]: k for k in class_to_idx}
    mapped_classes = list()
    # 这里要将这两个东西放回CPU 不然转NUMPY有问题！切记
    # 参考：https://blog.csdn.net/jizhidexiaoming/article/details/82423935?utm_source=blogxgwz5
    top_labels_np = top_labels.data.cpu().numpy()
    top_prob_np = top_prob_ori.data.cpu().numpy()
    # 遍历labels
    for label in top_labels_np[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob_np[0], mapped_classes


def main():

    parser = argparse.ArgumentParser(
        description='My Flower Classification Predictor')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Use gpu or not')
    parser.add_argument('--image_path', type=str, help='Path of image')
    parser.add_argument('--hidden_units', type=int,
                        default=512, help='Hidden units for fc layer')
    parser.add_argument('--saved_model', type=str,
                        default='my_checkpoint_cmd.pth', help='Path of your saved model')
    parser.add_argument('--mapper_json', type=str, default='cat_to_name.json',
                        help='Path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5,
                        help='Display Topk probabilities')

    args = parser.parse_args()

    import json
    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    model, class_to_idx, idx_to_class = load_checkpoint(args)
    top_probability, top_class = predict(
        args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)

    print('预测分类的索引为: ', top_class)
    print('花卉的类别名为: ')
    [print(cat_to_name[x]) for x in top_class]
    print('其概率分别为: ', top_probability)


if __name__ == "__main__":
    main()
