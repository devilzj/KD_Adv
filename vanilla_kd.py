import os

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm

from train_scratch import test
from models import ResNet18, ResNet34
from utils import get_dataset, save_checkpoint, prepare_folders
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)  # 蒸馏+交叉熵


# 训练蒸馏学生网络一个epoch
def kd_train(teacher_net, student_net, train_loader, optimizer):
    student_net.train()
    teacher_net.eval()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            s_output = student_net(data)
            t_output = teacher_net(data).detach()
            loss = distillation(s_output, target, t_output, temp=5.0, alpha=0.7)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(train_loader.dataset) * 100
            epochs.set_description(description.format(avg_loss, acc))
    return acc, avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--store_name', type=str, default='',
                        help='exp statement')
    parser.add_argument('--resume', type=bool, default=False,
                        help='load ckpt')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    prepare_folders(args.store_name)
    tf_writer = SummaryWriter(log_dir=args.store_name + '/logs', )

    train_loader, test_loader = get_dataset(args)
    teacher_net = ResNet34().cuda()
    teacher_net.load_state_dict(torch.load('./resnet34/ckpts/model.pkl')['state_dict'])
    teacher_net.eval()
    print("Teacher acc:")
    test(teacher_net, test_loader)
    student_net = ResNet18().cuda()

    optimizer = optim.SGD(student_net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    bst_acc = -1

    for epoch in tqdm(range(1, args.epochs + 1)):
        adjust_learning_rate(optimizer, epoch)
        train_acc, train_loss = kd_train(teacher_net, student_net, train_loader, optimizer)
        test_acc, test_loss = test(student_net, test_loader)
        tf_writer.add_scalar('train_loss', train_loss, epoch)
        tf_writer.add_scalar('train_accuracy', train_acc, epoch)

        tf_writer.add_scalar('test_loss', test_loss, epoch)
        tf_writer.add_scalar('test_accuracy', test_acc, epoch)

        is_best = test_acc > bst_acc
        bst_acc = max(bst_acc, test_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_net.state_dict(),
            'best_acc': float(bst_acc),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.store_name + '/ckpts/model.pkl')

"""
CUDA_VISIBLE_DEVICES=2 python3 vanilla_kd.py --store_name=resnet34_resnet18

"""
