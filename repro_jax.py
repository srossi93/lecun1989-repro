"""
Running this script eventually gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm 
import jax 
import jax.numpy as jnp 
from typing import NamedTuple 
import optax

class Params(NamedTuple):
    H1w : jnp.array
    H1b : jnp.array
    H2w : jnp.array
    H2b : jnp.array
    H3w : jnp.array
    H3b : jnp.array
    outw: jnp.array 
    outb: jnp.array  
# -----------------------------------------------------------------------------


def init(rng):
    # initialization as described in the paper to my best ability, but it doesn't look right...
    winit = lambda rng, fan_in, *shape: (jax.random.uniform(rng, shape) - 0.5) * 2 * 2.4 / fan_in**0.5
    rng1, rng2, rng3, rngout = jax.random.split(rng, 4)

    # H1 layer parameters and their initialization
    H1w = winit(rng1, 5*5*1, 12, 1, 5, 5)
    H1b = jnp.zeros((12, 8, 8)) # presumably init to zero for biases

    # H2 layer parameters and their initialization
    H2w = winit(rng2, 5*5*8, 12, 8, 5, 5)
    H2b = jnp.zeros((12, 4, 4)) # presumably init to zero for biases

    # H3 is a fully connected layer
    H3w = winit(rng3, 4*4*12, 4*4*12, 30)
    H3b = jnp.zeros(30)

    # output layer is also fully connected layer
    outw = winit(rngout, 30, 30, 10)
    outb = -jnp.ones(10) # 9/10 targets are -1, so makes sense to init slightly towards it
    
    params = Params(H1w, H1b, H2w, H2b, H3w, H3b, outw, outb)
    return params

@jax.jit 
def forward(params, x):

    (H1w, H1b, H2w, H2b, H3w, H3b, outw, outb) = params
    # x has shape (1, 1, 16, 16)
    #  x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
    x = jnp.pad(x, ((0,0), (0,0), (2,2), (2,2)), 'constant', constant_values=-1.0)
    #  x = F.conv2d(x, H1w, stride=2) + H1b
    x = jax.lax.conv_general_dilated(x, H1w, padding="VALID", window_strides=(2,2)) + H1b
    x = jax.nn.tanh(x)

    # x is now shape (1, 12, 8, 8)
    x = jnp.pad(x, ((0,0), (0,0), (2,2), (2,2)), 'constant', constant_values=-1.0)
    slice1 = jax.lax.conv_general_dilated(x[:, 0:8], H2w[0:4], padding="VALID",window_strides=(2,2)) # first 4 planes look at first 8 input planes
    slice2 = jax.lax.conv_general_dilated(x[:, 4:12], H2w[4:8], padding="VALID",window_strides=(2,2)) # next 4 planes look at last 8 input planes
    slice3 = jax.lax.conv_general_dilated(jnp.concatenate((x[:, 0:4], x[:, 8:12]), axis=1), H2w[8:12], padding="VALID",window_strides=(2,2)) # last 4 planes are cross
    x = jnp.concatenate((slice1, slice2, slice3), axis=1) + H2b
    x = jnp.tanh(x)

    # x is now shape (1, 12, 4, 4)
    x = x.reshape(len(x), -1) # (1, 12*4*4)
    x = x @ H3w + H3b
    x = jnp.tanh(x)

    # x is now shape (1, 30)
    x = x @ outw + outb
    x = jnp.tanh(x)

     # x is finally shape (1, 10)
    return x

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))

    # init rng
    rng = jax.random.PRNGKey(0)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    #  writer = SummaryWriter(args.output_dir)

    # init a model
    params = init(rng)


    # init data
    Xtr, Ytr = torch.load('train1989.pt')
    Xte, Yte = torch.load('test1989.pt')
    Xtr, Ytr = Xtr.numpy(), Ytr.numpy()
    Xte, Yte = Xte.numpy(), Yte.numpy()

    # init optimizer
    init_opt, update = optax.sgd(learning_rate=args.learning_rate,) 
    state_opt = init_opt(params)

    def eval_split(params, split):
        # eval the full train/test set, batched implementation for efficiency
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = forward(params, X)
        loss = jnp.mean((Y - Yhat)**2)
        err = jnp.mean((jnp.argmax(Y, axis=1) != jnp.argmax(Yhat, axis=1)))
        print(f"eval: split {split:5s}. loss {loss:e}. error {err*100:.2f}%. misses: {int(err*len(Y))}")
        #  writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        #  writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

    eval_split(params, 'test')

    @jax.grad
    def grad_loss(params, x, y):
        yhat = forward(params, x)
        return jnp.mean((y-yhat)**2)
    
    @jax.jit
    def step(params, state_opt, x, y):
        grads = grad_loss(params, x, y)
        updates, state_opt = update(grads, state_opt)
        params = optax.apply_updates(params, updates)
        return params, state_opt

        
    step(params, state_opt, Xtr, Ytr)

    # train
    with tqdm(total=23*len(Xtr)) as pbar:
        for pass_num in (range(23)):
    
            # perform one epoch of training
            for step_num in (range(len(Xtr))):
    
                # fetch a single example into a batch of 1
                x, y = Xtr[[step_num]], Ytr[[step_num]]
                params, state_opt = step(params, state_opt, x, y)
                #  yhat = forward(params, x)
                pbar.update()
    

        # after epoch epoch evaluate the train and test error / metrics
    eval_split(params, 'train')
    eval_split(params, 'test')

    # save final model to file
