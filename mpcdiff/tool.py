if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from train_plaintext import LeNet
from train_plaintext import GloroTorch
from PIL import Image
import glob
import numpy as np
# from examples.util import NoopContextManager
# from examples.meters import AverageMeter
import crypten
import crypten.communicator as comm
import time, logging, tqdm, warnings
import crypten.mpc as mpc
from crypten.nn import model_counter
import multiprocessing, copy
from torchvision.utils import save_image
import shutil
import os
import joblib
torch.set_num_threads(1)

from fuzz import fuzzer
from fuzz_scikit import fuzzer_scikit

class Logistic(nn.Sequential):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(23, 120)
        self.fc2 = nn.Linear(120, 2)
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

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

def DT(batch_size, pool_size, guide, dataset, mpc_net, plain_net):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--pool_size', type=int, default=1000)
    # parser.add_argument('--guide', type=str, default='err')
    # parser.add_argument('--dataset', type=str, default='MNIST')
    # parser.add_argument('--test_only', action='store_true')
    # parser.add_argument('--repaired', action='store_true')
    # args = parser.parse_args()
    args = argparse.Namespace()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device
    args.batch_size = batch_size
    args.pool_size = pool_size
    args.guide = guide
    args.dataset = dataset
    args.test_only = False
    args.repaired = False

    if args.dataset == 'MNIST':
        out_dir = './results/' + args.dataset + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir) 
        os.mkdir(out_dir)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        crypten.init()
        model_counter.register_counter(args)
        from crypten.config import cfg
        net = mpc_net
        if args.test_only:
            file_name = './results/MNIST/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                cfg.encoder.precision_bits = bit
                mpc_fuzzer = fuzzer(args)
                acc = mpc_fuzzer.test_mpc(net, testloader, args)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f\n'%(bit, acc))
        else:
            mpc_fuzzer = fuzzer(args)
            mpc_fuzzer.start(net, testloader, args)

    elif args.dataset == 'Credit':
        out_dir = './results/' + args.dataset + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir) 
        os.mkdir(out_dir)
        data_list = np.load('./dataset/Credit/credit_card_clients_data.npy')
        label_list = np.load('./dataset/Credit/credit_card_clients_label.npy')
        data_tensor = torch.from_numpy(np.array(data_list))
        label_tensor = torch.from_numpy(np.array(label_list)).type(torch.LongTensor)
        test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 23))
        test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        crypten.init()
        model_counter.register_counter(args)
        from crypten.config import cfg
        net = mpc_net

        if args.test_only:
            file_name = './results/Credit/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                cfg.encoder.precision_bits = bit
                mpc_fuzzer = fuzzer(args)
                acc = mpc_fuzzer.test_mpc(net, testloader, args)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f\n'%(bit, acc))
        else:
            mpc_fuzzer = fuzzer(args)
            mpc_fuzzer.start(net, testloader, args)

    elif args.dataset == 'Bank':
        out_dir = './results/' + args.dataset + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir) 
        os.mkdir(out_dir)
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
    elif args.dataset == 'Bank_sci':
        out_dir = './results/' + args.dataset + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir) 
        os.mkdir(out_dir)
        data_list = np.load('./dataset/Bank/bank_data.npy')
        label_list = np.load('./dataset/Bank/bank_label.npy')
        test_data = data_list[-args.pool_size:].reshape((-1, 1, 20))
        test_label = label_list[-args.pool_size:].reshape((-1, 1))
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        mpc_fuzzer = fuzzer_scikit(args)
        mpc_fuzzer.start(mpc_net, testloader, args, plain_net)