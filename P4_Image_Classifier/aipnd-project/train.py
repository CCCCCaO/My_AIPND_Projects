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
from torchvision import datasets, models, transforms, utils

# 判断GPU是否可用 清楚GPU缓存
use_gpu = torch.cuda.is_available()
torch.cuda.empty_cache()

def cook_data(args):
    '''
    加载并预处理数据
    输入参数args为train valid test的根路径
    '''
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # 一些变换
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # 用ImageFolder加载数据集
    image_datasets = dict()  # 建立个字典更方便后面操作
    image_datasets['train'] = datasets.ImageFolder(
        train_dir, transform=train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(
        valid_dir, transform=valid_transforms)
    image_datasets['test'] = datasets.ImageFolder(
        test_dir, transform=test_transforms)
    # 加载数据 指定batch_size 这里是10
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=32, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=32, shuffle=True)
    dataloaders['test'] = torch.utils.data.DataLoader(
        image_datasets['test'], batch_size=32, shuffle=True)

    return dataloaders, image_datasets


def train_model(args, model, criterion, optimizer, num_epochs=8):
    '''
    训练模型
    '''
    dataloaders, image_datasets = cook_data(args)
    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'valid', 'test']}
    since = time.time()
    # 深拷贝一份 避免更改原始的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    print_every = 100
    steps = 0
    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e+1, num_epochs))
        print('-' * 10)
        model.train()
        running_loss = 0
        # 训练
        for data in dataloaders['train']:
            inputs, labels = data
            steps += 1
            # 转换成Variable
            if use_gpu and args.gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 计算损失
            running_loss += loss.item()
            # 打印结果
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, num_epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                # 损失置零
                running_loss = 0
        print()
        # 用验证集在每个epoch结束后来验证一下准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # 不反向传播梯度
            for data in dataloaders['valid']:
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                outputs_valid = model(inputs)
                _, predicted = torch.max(outputs_valid.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = (100.0 * correct / total)
            print("本epoch结束后，验证准确率为：%.2f %% \n" % acc)
            print()

    time_elapsed = time.time() - since
    print('训练耗时： {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # 加载模型
    model.load_state_dict(best_model_wts)
    return model


def train_model_wrapper(args):
    dataloaders, image_datasets = cook_data(args)
    # 加载预先训练好的神经网络 vgg16 或 vgg13
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    # 冻结参数
    for param in model.parameters():
        param.requires_grad = False
    # 设计一个新的分类器 可以自己输入隐藏
    num_features = model.classifier[0].in_features  # VGG
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
    # 更换分类器
    model.classifier = classifier

    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print("是否使用GPU: " + str(use_gpu))
        else:
            print("由于GPU配置有问题，使用CPU")
    # 定义误差函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    # 调用训练函数
    model = train_model(args, model, criterion,
                        optimizer, num_epochs=args.epochs)
    # 获取标签映射
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    # 保存检查点 以便下次使用 参数是手工输入的路径saved_model
    model.epochs = args.epochs
    checkpoint = {'input_size': [3, 224, 224],
                  'batch_size': dataloaders['train'].batch_size,
                  'output_size': 102,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    torch.save(checkpoint, args.saved_model)


def main():
    parser = argparse.ArgumentParser(
        description='My Flower Classifcation trainer')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='architecture [available: vgg13, vgg16]')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int,
                        default=512, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--data_dir', type=str,
                        default='flowers', help='dataset directory')
    parser.add_argument('--saved_model', type=str,
                        default='my_checkpoint_cmd.pth', help='path of your saved model')
    args = parser.parse_args()

    train_model_wrapper(args)


if __name__ == "__main__":
    main()
