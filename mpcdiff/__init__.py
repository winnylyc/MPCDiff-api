#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
# from GenerateMPCAE import LeNet
from mpcdiff.train_plaintext import LeNet
from mpcdiff.train_plaintext import GloroTorch
from PIL import Image
import glob
import numpy as np
# from examples.util import NoopContextManager
# from examples.meters import AverageMeter
import mpcdiff.crypten
import mpcdiff.crypten.communicator as comm
import time, logging, tqdm, warnings
import mpcdiff.crypten.mpc as mpc
from mpcdiff.crypten.nn import model_counter
import multiprocessing, copy
from torchvision.utils import save_image
from mpcdiff.fuzz import fuzzer
torch.set_num_threads(1)

class BankNet(nn.Sequential):
    def __init__(self):
        super(BankNet, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class BankNetMPC(nn.Sequential):
    def __init__(self):
        super(BankNetMPC, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.ELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def fuzztest():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=1000)
    parser.add_argument('--guide', type=str, default='err')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--repaired', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device
    args.model_path = './model/Bank.pth'
    data_list = np.load('./dataset/Bank/bank_data.npy')
    label_list = np.load('./dataset/Bank/bank_label.npy')
    data_tensor = torch.from_numpy(data_list)
    label_tensor = torch.from_numpy(label_list).type(torch.LongTensor)
    test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 20))
    test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
    testloader = []
    for i in range(test_data.shape[0]):
        testloader.append((test_data[i], test_label[i]))

    crypten.init()
    model_counter.register_counter(args)
    from crypten.config import cfg
    mpc_net = BankNetMPC()
    mpc_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    mpc_net.eval()

    plain_net = BankNet()
    plain_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    plain_net.eval()

    if args.test_only:
        file_name = './results/Bank/precision_bit.txt'
        precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        for bit in precision_bits:
            print('current bit', bit)
            cfg.encoder.precision_bits = bit
            mpc_fuzzer = fuzzer(args)
            acc = mpc_fuzzer.test_mpc(mpc_net, testloader, args)
            with open(file_name, 'a') as txt_file:
                txt_file.write('%d, \t%f\n'%(bit, acc))
    else:
        mpc_fuzzer = fuzzer(args)
        mpc_fuzzer.start(mpc_net, testloader, args, plain_net)