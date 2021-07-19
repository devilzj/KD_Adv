import os

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm

from models import ResNet18, ResNet34
from utils import get_dataset, save_checkpoint, prepare_folders
from torch.utils.tensorboard import SummaryWriter

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


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            optimizer.zero_grad()
            epochs.set_description(description.format(avg_loss, acc))
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(train_loader.dataset) * 100
    return acc, avg_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss




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
    tf_writer = SummaryWriter(log_dir=args.store_name+'/logs', )

    train_loader, test_loader = get_dataset(args)
    model = ResNet34().cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loading checkpoint from epoch:{}, original acc:{}, "
                  .format(args.start_epoch, checkpoint['best_acc'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    bst_acc = -1

    for epoch in tqdm(range(1, args.epochs + 1)):
        adjust_learning_rate(optimizer, epoch)
        train_acc, train_loss = train(model, train_loader, optimizer)
        test_acc, test_loss = test(model, test_loader)
        tf_writer.add_scalar('train_loss', train_loss, epoch)
        tf_writer.add_scalar('train_accuracy', train_acc, epoch)

        tf_writer.add_scalar('test_loss', test_loss, epoch)
        tf_writer.add_scalar('test_accuracy', test_acc, epoch)

        is_best = test_acc > bst_acc
        bst_acc = max(bst_acc, test_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': float(bst_acc),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.store_name+'/ckpts/model.pkl')

"""
python3 train_scratch.py 

"""