from __future__ import print_function
from __future__ import print_function
import random
import math
import argparse
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from centerloss import ImprovedCenterLoss, CenterLoss

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('expname', type=str,
                    help='name of the experiment')
parser.add_argument('--batch-size', type=int, default=100, 
                    help='input batch size for training (default: 64)')
parser.add_argument('--batches-per-epoch', type=int, default=40,
                    help='number of batches per epoch')
parser.add_argument('--epochs', type=int, default=1000, 
                    help='number of epochs to train (default: 100)')
parser.add_argument('--labeled-train-size', type=int, default=4000,
                    help='labeled training dataset size for training')
parser.add_argument('--unlabeled-train-size', type=int, default=36000,
                    help='unlabeled training dataset size for training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = datasets.CIFAR10('../data', train=True, download=True)

class Dataset(object):

    def __init__(self, base_dataset,
                 transform=transforms.Compose([
                     transforms.Resize((32, 32)),
                     transforms.ToTensor()])):
    
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, i):
        x, y = self.base_dataset[i]
        return self.transform(x), y
        

labeled_train_dataset, unlabeled_train_dataset = \
         train_test_split(train_dataset,
                          train_size=args.labeled_train_size,
                          test_size=args.unlabeled_train_size,
                          random_state=args.seed)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop((32, 32), padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

labeled_train_dataset = Dataset(labeled_train_dataset, transform)
unlabeled_train_dataset = Dataset(unlabeled_train_dataset, transform)

print('#labeled, #unlabeled =', len(labeled_train_dataset), len(unlabeled_train_dataset))

labeled_train_loader = torch.utils.data.DataLoader(
    labeled_train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
cycle_labeled_train_loader = itertools.cycle(labeled_train_loader)

unlabeled_train_loader = torch.utils.data.DataLoader(
    unlabeled_train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
cycle_unlabeled_train_loader = itertools.cycle(unlabeled_train_loader)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False,
                     transform=transforms.Compose([
                         transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                     ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, leak=0.1):
        super(Conv2dBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leak = leak

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(leak),
        )
        
    def forward(self, x):
        return self.main(x)



class ConvLarge(nn.Module):

    def __init__(self):
        super(ConvLarge, self).__init__()

        self.net = nn.Sequential(
            Conv2dBlock(3, 128, 3, 1, 1),            
            Conv2dBlock(128, 128, 3, 1, 1),
            Conv2dBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.5),
            Conv2dBlock(128, 256, 3, 1, 1),
            Conv2dBlock(256, 256, 3, 1, 1),
            Conv2dBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.5),
            Conv2dBlock(256, 512, 3, 1, 1),
            Conv2dBlock(512, 256, 1, 1, 0),
            Conv2dBlock(256, 128, 1, 1, 0),
            nn.AvgPool2d(8, 1),
        )
        
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x # return logit

class NullCenterLoss(nn.Module):
    def __init__(self):
        super(NullCenterLoss, self).__init__()
        
    def forward(self, *x):

        if x[0].is_cuda:
            return Variable(torch.FloatTensor([0]).cuda())
        else:
            return Variable(torch.FloatTensor([0]))

    
model = ConvLarge()
#centerloss = ImprovedCenterLoss(10, 0.6)
centerloss = CenterLoss(10, 10, 0.6)
#centerloss = NullCenterLoss()
if args.cuda:
    model.cuda()
    centerloss.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer = SummaryWriter(comment='-{}'.format(args.expname))

def train(epoch):
    alpha = 0.1
    model.train()
    for batch_idx in range(args.batches_per_epoch):
        data, target = next(cycle_labeled_train_loader)
#    for batch_idx, (data, target) in enumerate(labeled_train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        model.zero_grad()
        logit = model(data)
        logprob = F.log_softmax(logit)
        if type(model) == CenterLoss:
            closs = centerloss(logit, target)
        else:
            closs = centerloss(logprob, target)
        loss = F.nll_loss(logprob, target)
        (loss + alpha * closs).backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'
                  '\tLoss: {:.6f} (loss: {:.6f}, closs: {:.6f})'.format(
                epoch, batch_idx * len(data), len(labeled_train_loader.dataset),
                      100. * batch_idx / len(labeled_train_loader),
                      (loss+0.1*closs).data[0], loss.data[0], closs.data[0]))
            n = epoch * len(labeled_train_loader.dataset) + batch_idx * args.batch_size
            writer.add_scalar('cifar10/train-loss', (loss + 0.1*closs).data[0], n)
            writer.add_scalar('cifar10/train-xent-loss', loss.data[0], n)
            writer.add_scalar('cifar10/train-center-loss', closs.data[0], n)
            

def test(epoch):
    model.eval()
    centerloss.eval()
    test_loss = 0
    test_closs = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        logit = model(data)
        logprob = F.log_softmax(logit)
        if type(centerloss) == CenterLoss:
            test_closs += centerloss(logit, target).data[0]
        else:
            test_closs += centerloss(logprob, target).data[0]            
        test_loss += F.nll_loss(logprob, target).data[0]

        pred = logprob.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader) # loss function already averages over batch size
    test_closs /= len(test_loader) # loss function already averages over batch size
    print('Test set: Average loss: {:.4f}, xent-loss: {:.4f}, closs: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        (test_loss + 0.1 * test_closs),
        test_loss, test_closs, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    writer.add_scalar('cifar10/test-accuracy', correct / len(test_loader.dataset), epoch)
    writer.add_scalar('cifar10/test-loss', (test_loss + 0.1*test_closs), epoch)
    writer.add_scalar('cifar10/test-xent-loss', test_loss, epoch)
    writer.add_scalar('cifar10/test-center-loss', test_closs, epoch)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

writer.close()
