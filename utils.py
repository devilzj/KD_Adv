import os

import torch
import torchvision.transforms as transforms
from torchvision import datasets


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)



def prepare_folders(store_name):
    if not os.path.exists(store_name):
        print('creating folder ' + store_name)
        os.mkdir(store_name)
        os.mkdir(store_name+'/logs')
        os.mkdir(store_name + '/ckpts')

def get_dataset(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader
