from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags
from model import Attention, GatedAttention
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser(description='PyTorch H&E bags Example')

parser.add_argument('--feature_path', type=str, default='', help='path for features')
parser.add_argument('--y_path', type=str, default='', help='path for y')
parser.add_argument('--model_path', type=str, default='', help='path for model')
parser.add_argument('--graph_path', type=str, default='', help='path for graphs')

parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--batch_size', type=int, default=1, help='number of slides in a batch')                  
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--early_stopping', type=int, default=4, help='epochs until stopping')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')


def get_y(list_samples, args):

    dict_y = pd.read_csv(os.path.join(args.y_path, "train_output.csv")).to_dict()
    dict_y = {dict_y['Sample ID'][key] : dict_y['Target'][key] for key in dict_y['Sample ID'].keys()}
    y = np.zeros(len(list_samples))

    for i, sample in enumerate(list_samples):
        y[i] = dict_y[sample]
    
    return y


def get_loaders(args, list_samples, y, train_index, val_index):

    samples_train = [list_samples[i] for i in train_index]
    samples_val = [list_samples[i] for i in val_index]

    X_train = np.zeros((len(samples_train), 1000, 2048))
    X_val = np.zeros((len(samples_val), 1000, 2048))

    for i, sample in enumerate(samples_train):
        X_train[i] = np.load(os.path.join(args.feature_path, sample))[:, 3:]

    for i, sample in enumerate(samples_val):
        X_val[i] = np.load(os.path.join(args.feature_path, sample))[:, 3:]

    y_train = y[train_index]
    y_val = y[val_index]

    X_train = torch.Tensor(X_train) # transform to torch tensor
    y_train = torch.Tensor(y_train)

    dataset_train = TensorDataset(X_train, y_train) # create your datset
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True) # create your dataloader

    X_val = torch.Tensor(X_val) # transform to torch tensor
    y_val = torch.Tensor(y_val)

    dataset_val = TensorDataset(X_val, y_val) # create your datset
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True) # create your dataloader
    
    return train_loader, val_loader


def train(epoch):
    model.train()
    train_loss = 0.
    list_label = []
    list_pred = []
    for batch_idx, (data, label) in enumerate(train_loader):
        # bag_label = label[0]
        bag_label = label

        list_label.append(int(bag_label.detach().numpy()[0]))
        # list_label += list(bag_label.detach().numpy())
        print(list_label[-1])

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _, y_prob = model.calculate_objective(data, bag_label)
        print(loss.data)
        train_loss += loss.data[0]
        # train_loss += loss.item()

        list_pred.append(y_prob.detach().numpy()[0, 0])
        # list_pred += list(y_prob.detach().numpy()[:,0, 0])
        print(list_pred[-1])

        # backward pass
        loss.backward()
        # step
        optimizer.step()

        assert 1==0

    # calculate loss and error for epoch
    train_loss /= len(train_loader)

    auc_epoch = roc_auc_score(list_label, list_pred)

    return train_loss.cpu().numpy()[0], auc_epoch
    # return train_loss, auc_epoch


def eval(epoch):
    model.eval()
    val_loss = 0.
    list_label = []
    list_pred = []
    for batch_idx, (data, bag_label) in enumerate(val_loader):
        list_label.append(int(bag_label.numpy()[0]))
        # list_label += list(bag_label.detach().numpy())

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # calculate loss and metrics
        loss, _, y_prob = model.calculate_objective(data, bag_label)
        val_loss += loss.data[0]
        # val_loss += loss.item()

        list_pred.append(y_prob.detach().numpy()[0, 0])
        # list_pred += list(y_prob.detach().numpy()[:,0, 0])
     
    # calculate loss and error for epoch
    val_loss /= len(val_loader)
    auc_epoch = roc_auc_score(list_label, list_pred)

    return val_loss.cpu().numpy()[0], auc_epoch
    # return val_loss, auc_epoch


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    list_samples = os.listdir(args.feature_path)
    y = get_y(list_samples, args)

    list_auc = []

    new_model_path = os.path.join(args.model_path, str(args.batch_size) + '_' + str(args.lr) + '_' + str(args.reg))

    while os.path.exists(new_model_path):
        new_model_path = new_model_path + '_'

    os.makedirs(new_model_path)

    for i, (train_index, val_index) in enumerate(skf.split(list_samples, y)):
        print(f"\nSplit {i+1}:")

        print('Init Model')
        if args.model=='attention':
            model = Attention()
        elif args.model=='gated_attention':
            model = GatedAttention()
        if args.cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

        print('Load Train and Test Set')
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader, val_loader = get_loaders(args, list_samples, y, train_index, val_index)

        print('\nSTARTING TRAINING')

        list_loss_train = []
        list_auc_train = []

        list_loss_val = []
        list_auc_val = []

        auc_max = -1
        stop = 0
        message_es = True

        for epoch in range(1, args.epochs + 1):
            if stop < args.early_stopping:
                train_loss, train_auc = train(epoch)
                val_loss, val_auc = eval(epoch)

                if val_auc > auc_max:
                    auc_max = val_auc
                    torch.save(model.state_dict(), os.path.join(new_model_path, 'model_fold_{}.pth'.format(i+1)))
                    stop = 0
                else:
                    stop += 1

                print('\nEpoch {}/{}'.format(epoch, args.epochs))
                print('Train Loss: {:.4f}, Val loss: {:.4f}'.format(train_loss, val_loss))
                print('Train AUC: {:.4f}, Val AUC: {:.4f}'.format(train_auc, val_auc))

                list_loss_train.append(train_loss)
                list_auc_train.append(train_auc)

                list_loss_val.append(val_loss)
                list_auc_val.append(val_auc)
            else:
                if message_es:
                    print('\nSTOPPING TRAINING')
                    message_es = False

        list_auc.append(auc_max)

        plt.plot(list_loss_train, label='train')
        plt.plot(list_loss_val, label='val')
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(args.graph_path, 'loss_fold_{}.png'.format(i+1)))
        plt.figure().clear()

        plt.plot(list_auc_train, label='train')
        plt.plot(list_auc_val, label='val')
        plt.legend()
        plt.title('AUC (best={:.3f})'.format(auc_max))
        plt.savefig(os.path.join(args.graph_path, 'auc_fold_{}.png'.format(i+1)))
        plt.figure().clear()
    
    print('\nAUC results\n')
    for i, auc in enumerate(list_auc):
        print('Fold {} : {:.3f}'.format(i+1, auc))