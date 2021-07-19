'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import attacks
import torch.nn as nn
from models import *
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from train_scratch import test
from utils import get_dataset, prepare_folders
from vanilla_kd import distillation
import warnings

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight_decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--epsilon', default=0.031,
                        help='perturbation')
    parser.add_argument('--num_steps', default=10,
                        help='perturb number of steps')
    parser.add_argument('--step_size', default=0.007,
                        help='perturb step size')
    parser.add_argument('--beta', default=6.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', default="Resnet34", type=str,
                        help='save frequency')
    parser.add_argument('--type', default="normal", type=str,
                        help='save frequency')
    parser.add_argument('--store_name', type=str, default='',
                        help='exp statement')

    args = parser.parse_args()
    return args


# Training
def train_attack_KD(t_net, s_net, args, train_loader):
    epoch_start_time = time.time()
    print('\nStage 1 Epoch: %d' % epoch)
    s_net.train()
    t_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    attack_size = 64
    temperature = 3
    description = "loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for batch_idx, (data, target) in enumerate(epochs):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            s_output = student_net(data)
            t_output = teacher_net(data).detach()
            loss = distillation(s_output, target, t_output, temp=5.0, alpha=0.7)

            if args.attack_ratio > 0:

                condition1 = target.data == t_output.sort(dim=1, descending=True)[1][:, 0].data
                condition2 = target.data == s_output.sort(dim=1, descending=True)[1][:, 0].data

                attack_flag = condition1 & condition2

                if attack_flag.sum():
                    # Base sample selection
                    attack_idx = attack_flag.nonzero().squeeze()
                    if attack_idx.shape[0] > attack_size:
                        diff = (F.softmax(t_output[attack_idx, :], 1).data - F.softmax(s_output[attack_idx, :],
                                                                                       1).data) ** 2
                        distill_score = diff.sum(dim=1) - diff.gather(1, target[attack_idx].data.unsqueeze(1)).squeeze()
                        attack_idx = attack_idx[distill_score.sort(descending=True)[1][:attack_size]]

                    # Target class sampling
                    attack_class = t_output.sort(dim=1, descending=True)[1][:, 1][attack_idx].data
                    class_score, class_idx = F.softmax(t_output, 1)[attack_idx, :].data.sort(dim=1, descending=True)
                    class_score = class_score[:, 1:]
                    class_idx = class_idx[:, 1:]

                    rand_seed = 1 * (class_score.sum(dim=1) * torch.rand([attack_idx.shape[0]]).cuda()).unsqueeze(1)
                    prob = class_score.cumsum(dim=1)
                    for k in range(attack_idx.shape[0]):
                        for c in range(prob.shape[1]):
                            if (prob[k, c] >= rand_seed[k]).cpu().numpy():
                                attack_class[k] = class_idx[k, c]
                                break

                    # Forward and backward for adversarial samples
                    attacked_inputs = Variable(attack.run(t_net, data[attack_idx, :, :, :].data, attack_class))
                    batch_size2 = attacked_inputs.shape[0]

                    attack_out_t = t_net(attacked_inputs)
                    attack_out_s = s_net(attacked_inputs)

                    # KD loss for Boundary Supporting Samples (BSS)
                    loss += - args.attack_ratio * (F.softmax(attack_out_t / temperature, 1).detach() * F.log_softmax(
                        attack_out_s / temperature, 1)).sum() / batch_size2

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(s_output[0:data.shape[0], :].data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().float().sum()
            epochs.set_description(description.format(train_loss / (batch_idx + 1), 100. * correct / total))


if __name__ == '__main__':
    args = get_args()
    train_loader, test_loader = get_dataset(args)
    prepare_folders(args.store_name)
    tf_writer = SummaryWriter(log_dir=args.store_name + '/logs', )

    teacher_net = ResNet34().cuda()
    teacher_net.load_state_dict(torch.load('./resnet34/ckpts/model.pkl')['state_dict'])
    student_net = ResNet18().cuda()
    print("Teacher acc:")
    test(teacher_net, test_loader)
    # Proposed adversarial attack algorithm (BSS)
    attack = attacks.AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, cuda=True, norm=2)

    criterion_MSE = nn.MSELoss(size_average=False)
    criterion_CE = nn.CrossEntropyLoss()
    max_epoch = 80
    for epoch in range(1, max_epoch + 1):
        if epoch == 1:
            optimizer = optim.SGD(student_net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        elif epoch == max_epoch / 2:
            optimizer = optim.SGD(student_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        elif epoch == max_epoch / 4 * 3:
            optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        args.attack_ratio = max(2 * (1 - 4 / 3 * epoch / max_epoch), 0) + 0

        train_attack_KD(teacher_net, student_net, args, train_loader)

        test(student_net, test_loader)


"""
--------------------------------------------------
Teacher  |  Student  | T_acc  | S_acc |   Type   |
--------------------------------------------------
Resnet34 | Resnet18  |  94.96 | 94.92 | Vanilla_kd
--------------------------------------------------


"""