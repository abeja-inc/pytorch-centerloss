import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from models import ConvLarge
from ema import EMA


def load_option():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str,
                        help='name of the experiment')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batches-per-epoch', type=int, default=-1,
                        help='number of batches per epoch if negative value is given use #dataset/batch_size')
    parser.add_argument('--max-epoch', type=int, default=300, 
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--labeled-train-size', type=int, default=4000,
                        help='labeled training dataset size for training')
    parser.add_argument('--unlabeled-train-size', type=int, default=46000,
                        help='unlabeled training dataset size for training if negative alue is given use all remaining data')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--regularization', type=str, choices=['TE', 'TE++', 'TE#', 'Null'], default='TE',
                        help='regularization type')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='decay rate of moving average')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='bayes inverse temparature (default=1.0)')
    # parser.add_argument('--unlabeled-label', type=int, default=-100, 
    #                     help='label for unlabeled data (default=-100)')
    parser.add_argument('--rampup-length', type=int, default=80,
                        help='duration linearly ramping-up beta')
    parser.add_argument('--rampdown-length', type=int, default=50,
                        help='duration linearly ramping-down learning rate')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes in dataset')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.unlabeled_label = args.num_classes
    return args


class SSDataset(object):

    def __init__(self, dataset, supervised_size, unsupervised_size=None, transform=None, unlabeled_label=-1):
        assert(supervised_size <= len(dataset)) 
        assert(unsupervised_size is None \
               or unsupervised_size < 0 \
               or supervised_size + unsupervised_size <= len(dataset))
        self.dataset = dataset
        self.supervised_size = supervised_size
        if unsupervised_size is not None and unsupervised_size >= 0:
            self.unsupervised_size = unsupervised_size
        else:
            self.unsupervised_size = len(dataset) - supervised_size
        self.transform = transform
        self.unlabeled_label = unlabeled_label

        indices = np.random.choice(np.arange(len(self.dataset)), len(self), replace=False)
        s_indices = np.random.choice(np.arange(len(self)), self.supervised_size, replace=False)
        supervised = np.zeros(len(self))
        supervised[s_indices] = 1
        assert((supervised == 1).sum() == self.supervised_size)
        assert((supervised == 0).sum() == self.unsupervised_size)
        self.indices = indices
        self.supervised = supervised  # supervised (1) or not unsupervised (0)

        assert(self.supervised_size >= 0)
        assert(self.unsupervised_size >= 0)

    def __len__(self):
        return self.supervised_size + self.unsupervised_size

    def __getitem__(self, i):
        idx = self.indices[i]
        s = self.supervised[i]
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        assert(y != self.unlabeled_label)
        if s:
            return i, x, y
        else:
            return i, x, self.unlabeled_label


class WhiteNoise(object):

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, x):
        n = x.clone()
        n.normal_(0, self.std)
        return x + n

    
def load_cifar10(opts, unlabeled_label):

    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    # compute the means and stds
    mean = np.asarray([train_dataset.train_data[:, :, :, i].mean() for i in range(3)])
    std = np.asarray([train_dataset.train_data[:, :, :, i].std() for i in range(3)])
    mean /= 255
    std /= 255
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32, 32), padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        WhiteNoise(0.15)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = SSDataset(train_dataset,
                              opts.labeled_train_size, opts.unlabeled_train_size,
                              transform=train_transform,
                              unlabeled_label=unlabeled_label)
    
    test_dataset = datasets.CIFAR10('./data', train=False,
                                    transform=test_transform)

    return train_dataset, test_dataset


class Trainer(object):
    
    def __init__(self, model, center, train_dataset, valid_dataset, opts):
        self.model = model
        self.center = center
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.opts = opts

        pin_memory = True if opts.cuda else False
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=opts.batch_size,
                                                            shuffle=True,
                                                            num_workers=1,
                                                            pin_memory=pin_memory)
        self.valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                            batch_size=opts.batch_size,
                                                            shuffle=False,
                                                            num_workers=1,
                                                            pin_memory=pin_memory)

        self.xent_loss = nn.CrossEntropyLoss(ignore_index=opts.unlabeled_label)
        self.optim = optim.Adam(self.model.parameters(), lr=opts.lr)
#        self.optim = optim.RMSprop(self.model.parameters(), lr=opts.lr)
        self.writer = SummaryWriter(comment='-{}'.format(opts.expname))
        self.epoch = 0
        self.n = 0

    @property
    def rampup(self):
        if self.epoch < self.opts.rampup_length:
            p = max(0, self.epoch / self.opts.rampup_length)
            p = 1.0 - p
            return math.exp(-p * p * 5.0)
        else:
            return 1.0

    @property
    def rampdown(self):
        if self.epoch >= (self.opts.max_epoch - self.opts.rampdown_length):
            ep = (self.epoch - (self.opts.max_epoch - self.opts.rampdown_length)) * 0.5
            return math.exp(-(ep * ep) / self.opts.rampdown_length)
        else:
            return 1.0

    def train(self):
        self.model.train()
        self.center.train()
        x = None
        y = None
        for batch_idx, (idx, data, label) in enumerate(self.train_dataloader):
            # set the learning rate
            beta = self.opts.beta * self.rampup
            lr = self.opts.lr * self.rampdown * self.rampup
            adam_beta = (self.rampdown * 0.9 + (1 - self.rampdown) * 0.5, 0.999)
            self.optim.param_groups[0]['lr'] = lr
            self.optim.param_groups[0]['beta'] = adam_beta
            # set the variables
            if x is None:
                assert(y is None)
                if self.opts.cuda:
                    data = data.cuda()
                    label = label.cuda()
                    idx = idx.cuda()
                x = Variable(data)
                y = Variable(label)
            else:
                assert(y is not None)
                x.data.resize_(data.size()).copy_(data)
                y.data.resize_(label.size()).copy_(label)
                if self.opts.cuda:
                    idx = idx.cuda()
            
            self.optim.zero_grad()
            logit = self.model(x)
            loss = self.xent_loss(logit, y)
            if self.opts.regularization == 'TE':
                prob = F.softmax(logit)                
                center = self.center(idx, prob)
                closs = ((prob - center) ** 2).mean()
            elif self.opts.regularization == 'TE++':
                logprob = F.log_softmax(logit)                
                center = self.center(idx, logprob)
                closs = ((logprob - center) ** 2).mean()
            elif self.opts.regularization == 'TE#':
                logprob = F.log_softmax(logit)                
                center = self.center(idx, logprob)
                closs = ((logprob[y.data] - center[y.data]) ** 2).mean()
#                closs = ((logprob - center) ** 2).mean()
            else:
                if self.opts.cuda:
                    closs = Variable(torch.FloatTensor([0]).cuda())                
                else:
                    closs = Variable(torch.FloatTensor([0]))                

            total_loss = loss + beta * closs
            total_loss.backward()
            self.optim.step()

            self.n += len(x.size())
            if batch_idx % self.opts.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'
                      '\ttotal_loss: {:.6f} (xent_loss: {:.6f}, center_loss: {:.6f})'.format(
                          self.epoch, batch_idx * self.opts.batch_size,
                          len(self.train_dataset),
                          100. * batch_idx / len(self.train_dataloader),
                          total_loss.data[0], loss.data[0], closs.data[0]))
                self.writer.add_scalar('cifar10/train-loss', total_loss.data[0], self.n)
                self.writer.add_scalar('cifar10/train-xent-loss', loss.data[0], self.n)
                self.writer.add_scalar('cifar10/train-center-loss', closs.data[0], self.n)
                
        self.epoch += 1
        
    def validate(self):
        self.model.eval()
        x = None
        y = None
        loss = 0
        correct = 0
        for batch_idx, (data, label) in enumerate(self.valid_dataloader):
            if x is None:
                assert(y is None)
                if self.opts.cuda:
                    data = data.cuda()
                    label = label.cuda()
                    x = Variable(data, volatile=True)
                    y = Variable(label, volatile=True)
            else:
                assert(y is not None)
                x.data.resize_(data.size()).copy_(data)
                y.data.resize_(label.size()).copy_(label)
            logit = self.model(x)
            prob = F.softmax(logit)
            logprob = torch.log(prob)
            loss += self.xent_loss(logit, y).data[0]
            y_pred = logit.data[:, :self.opts.num_classes].max(1)[1] # get the index with maximal probability
            correct += y_pred.eq(y.data).cpu().sum()
        loss /= len(self.valid_dataloader)
        print('Test Epoch: {}/{} ({:.1f}%)'
              '\txent-loss: {:.4f}, accuracy: {}/{} ({:.4f}%)'.format(
                  self.epoch,
                  self.opts.max_epoch,
                  100 * self.epoch / self.opts.max_epoch,
                  loss,
                  correct,
                  len(self.valid_dataset),
                  100 * correct / len(self.valid_dataset)))

        self.writer.add_scalar('cifar10/test-accuracy',
                               correct / len(self.valid_dataset), self.epoch)
        self.writer.add_scalar('cifar10/test-xent-loss', loss, self.epoch)

        
def main():

    opts = load_option()
    train_dataset, test_dataset = load_cifar10(opts, unlabeled_label=opts.unlabeled_label)

    model = ConvLarge(opts.num_classes + 1)
    center = EMA((len(train_dataset), opts.num_classes + 1), opts.alpha)
    if opts.cuda:
        model.cuda()
        center.cuda()

    trainer = Trainer(model, center, train_dataset, test_dataset, opts)

    trainer.validate()
    for epoch in range(opts.max_epoch):
        trainer.train()
        trainer.validate()
    
if __name__ == '__main__':
    main()
