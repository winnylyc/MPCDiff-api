from tool import DT

import argparse
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=1000)
    parser.add_argument('--guide', type=str, default='err')
    parser.add_argument('--dataset', type=str, default='MNIST')
    args = parser.parse_args()
    mpc_net = joblib.load('./model/ridge.model')
    plain_net = joblib.load('./model/lasso.model')
    DT(args.batch_size, args.pool_size, args.guide, args.dataset, mpc_net, plain_net)