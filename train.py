"""
Running this script eventually gives:
23
eval: split train. loss 3.993807e-03. error 0.60%. misses: 44
eval: split test . loss 2.837104e-02. error 4.04%. misses: 81
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # pip install tensorboard

# -----------------------------------------------------------------------------

class Net(nn.Module):
    """ 1989 LeCun ConvNet per description in the paper """

    def __init__(self):
        super().__init__()

        # H1 layer parameters and their initialization
        fan_in = 5*5*1
        up = 2.4 / fan_in**0.5
        w = torch.rand(12, 1, 5, 5) * up * 2 - up # U[-2.4 * F, 2.4 * F]
        self.H1w = nn.Parameter(w)
        self.H1b = nn.Parameter(torch.zeros(12, 8, 8)) # presumably init to zero for biases
        assert self.H1w.nelement() + self.H1b.nelement() == 1068

        # H2 layer parameters and their initialization
        """
        H2 neurons all connect to only 8 of the 12 input planes, with an unspecified pattern
        I am going to assume the most sensible block pattern where 4 planes at a time connect
        to differently overlapping groups of 8/12 input planes. We will implement this with 3
        separate convolutions that we concatenate the results of.
        """
        fan_in = 5*5*8
        up = 2.4 / fan_in**0.5
        w = torch.rand(12, 8, 5, 5) * up * 2 - up
        self.H2w = nn.Parameter(w)
        self.H2b = nn.Parameter(torch.zeros(12, 4, 4)) # presumably init to zero for biases
        assert self.H2w.nelement() + self.H2b.nelement() == 2592

        # H3 is a fully connected layer
        fan_in = 4*4*12
        up = 2.4 / fan_in**0.5
        w = torch.rand(fan_in, 30) * up * 2 - up
        self.H3w = nn.Parameter(w)
        self.H3b = nn.Parameter(torch.zeros(30))
        assert self.H3w.nelement() + self.H3b.nelement() == 5790

        # output layer is also fully connected layer
        fan_in = 30
        up = 2.4 / fan_in**0.5
        w = torch.rand(fan_in, 10) * up * 2 - up
        self.outw = nn.Parameter(w)
        self.outb = nn.Parameter(-torch.ones(10)) # 9/10 targets are -1, so makes sense to init slightly towards it
        assert self.outw.nelement() + self.outb.nelement() == 310

    def forward(self, x):

        # x has shape (1, 1, 16, 16)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = torch.tanh(x)

        # x is now shape (1, 12, 8, 8)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        slice1 = F.conv2d(x[:, 0:8], self.H2w[0:4], stride=2) # first 4 planes look at first 8 input planes
        slice2 = F.conv2d(x[:, 4:12], self.H2w[4:8], stride=2) # next 4 planes look at last 8 input planes
        slice3 = F.conv2d(torch.cat((x[:, 0:4], x[:, 8:12]), dim=1), self.H2w[8:12], stride=2) # last 4 planes are cross
        x = torch.cat((slice1, slice2, slice3), dim=1) + self.H2b
        x = torch.tanh(x)

        # x is now shape (1, 12, 4, 4)
        x = x.flatten(start_dim=1) # (1, 12*4*4)
        x = x @ self.H3w + self.H3b
        x = torch.tanh(x)

        # x is now shape (1, 30)
        x = x @ self.outw + self.outb
        x = torch.tanh(x)

         # x is finally shape (1, 10)
        return x

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
args = parser.parse_args()
print(vars(args))

# inits
torch.manual_seed(1337)
np.random.seed(1337)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)
writer = SummaryWriter(args.output_dir)

# init a model
model = Net()
print("number of params: ", sum(p.numel() for p in model.parameters())) # in paper total is 9,760

# init data
Xtr, Ytr = torch.load('train1989.pt')
Xte, Yte = torch.load('test1989.pt')

# init optimizer
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

def eval_split(split):
    # eval the full train/test set, batched implementation for efficiency
    model.eval()
    X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
    Yhat = model(X)
    loss = torch.mean((Y - Yhat)**2)
    err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
    print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
    writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
    writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

# train
for pass_num in range(23): # TODO: training longer helps

    # perform one epoch of training
    model.train()
    for step_num in range(Xtr.size(0)):

        # fetch a single example into a batch of 1
        x, y = Xtr[[step_num]], Ytr[[step_num]]

        # forward the model and the loss
        yhat = model(x)
        loss = torch.mean((y - yhat)**2)

        # calculate the gradient and update the parameters
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # after epoch epoch evaluate the train and test error / metrics
    print(pass_num + 1)
    eval_split('train')
    eval_split('test')
