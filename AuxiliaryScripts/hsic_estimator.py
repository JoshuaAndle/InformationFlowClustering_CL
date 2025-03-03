import torch
# import networks as net
# import dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from itertools import islice
import torch.utils.data as D
from tqdm import tqdm
import numpy as np
import math

import time


import os
import argparse
import json


def sigma_estimation(X, Y):
    """ sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med

def distmat(X):
    """ distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    X.view(1, -1)
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma):
    """ kernel matrix baker"""
    m = int(X.size()[0])
    # dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2. * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))

    Kxc = torch.mm(Kx, H)

    return Kxc

def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1E-5         #
    K_I = torch.eye(m)     #
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)  #
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)  #
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy






    
def cka_computation(x,y):
    x = x.cuda()
    y = y.cuda()
    
    batchsize = 128

    batches_cka = []
    xs = x.size(1)
    ys = y.size(1)
    print("Getting CKA for x and y of shapes: ", x.size(), " ", y.size(), " Batch size: ", batchsize, " xs and ys length: ", xs, " ", ys, flush=True)
    batches = int(math.ceil(y.size(0)/batchsize))
    for i in range(batches):
        if i != (batches - 1):
            xb = x[i*batchsize:(i+1)*batchsize]
            yb = y[i*batchsize:(i+1)*batchsize]
        else:
            xb = x[i*batchsize:]
            yb = y[i*batchsize:]
        
        for xi in range(xs):
            for yj in range(ys):
                xbi = torch.unsqueeze(xb[:,xi],dim=1)
                ybj = torch.unsqueeze(yb[:,yj],dim=1)
                batch_hsic = hsic_normalized_cca(xbi, ybj, sigma=2)
                batch_cka = batch_hsic / np.sqrt(hsic_normalized_cca(xbi,xbi, sigma=2)* hsic_normalized_cca(ybj,ybj, sigma=2))

                batches_cka.append(batch_cka)
    print("Batches CKA for layer with len ", len(batches_cka), ": ", batches_cka, flush=True)

    return torch.mean(torch.FloatTensor(batches_cka))



def hsic_computation(x,y):
    # name of layers

    x = x.cuda()
    y = y.cuda()
    
    batchsize = 128

    batches_hsic = []
    xs = x.size(1)
    ys = y.size(1)
    print("Getting HSIC for x and y of shapes: ", x.size(), " ", y.size(), " Batch size: ", batchsize, " xs and ys length: ", xs, " ", ys, flush=True)
    batches = int(math.ceil(y.size(0)/batchsize))
    for i in range(batches):
        if i != (batches - 1):
            xb = x[i*batchsize:(i+1)*batchsize]
            yb = y[i*batchsize:(i+1)*batchsize]
        else:
            xb = x[i*batchsize:]
            yb = y[i*batchsize:]
        
        for xi in range(xs):
            for yj in range(ys):
                xbi,ybj = torch.unsqueeze(xb[:,xi],dim=1), torch.unsqueeze(yb[:,yj],dim=1)
                batch_hsic = hsic_normalized_cca(xbi, ybj, sigma=2)

                batches_hsic.append(batch_hsic)
    return torch.mean(torch.FloatTensor(batches_hsic))


def hsic_and_cka_computation(x,y):
    # name of layers
    x = x.cuda()
    y = y.cuda()
    
    batchsize = 128

    batches_hsic = []
    batches_cka = []
    xs = x.size(1)
    ys = y.size(1)
    print("Getting HSIC and CKA for x and y of shapes: ", x.size(), " ", y.size(), " Batch size: ", batchsize, " xs and ys length: ", xs, " ", ys, flush=True)
    batches = int(math.ceil(y.size(0)/batchsize))
    for i in range(batches):
        if i != (batches - 1):
            xb = x[i*batchsize:(i+1)*batchsize]
            yb = y[i*batchsize:(i+1)*batchsize]
        else:
            xb = x[i*batchsize:]
            yb = y[i*batchsize:]
        
        for xi in range(xs):
            xbi = torch.unsqueeze(xb[:,xi],dim=1)
            batch_hsic = hsic_normalized_cca(xbi, yb, sigma=2)
            batch_cka = batch_hsic / np.sqrt(hsic_normalized_cca(xbi,xbi, sigma=2)* hsic_normalized_cca(yb,yb, sigma=2))

            batches_hsic.append(batch_hsic)
            batches_cka.append(batch_cka)
    print("Batches HSIC and CKA for layer with len ", len(batches_hsic), " ",len(batches_cka), flush=True)

    return torch.mean(torch.FloatTensor(batches_hsic)), torch.mean(torch.FloatTensor(batches_cka))




def hsic_and_cka_layers_computation(x,y):
    # name of layers
    x = x.cuda()
    y = y.cuda()
    
    batchsize = 512

    batches_hsic = []
    batches_cka = []
    xs = x.size(1)
    ys = y.size(1)
    print("Getting HSIC and CKA for x and y of shapes: ", x.size(), " ", y.size(), " Batch size: ", batchsize, " xs and ys length: ", xs, " ", ys, flush=True)
    batches = int(math.ceil(y.size(0)/batchsize))
    for i in range(batches):
        if i != (batches - 1):
            xb = x[i*batchsize:(i+1)*batchsize]
            yb = y[i*batchsize:(i+1)*batchsize]
        else:
            xb = x[i*batchsize:]
            yb = y[i*batchsize:]
        

            batch_hsic = hsic_normalized_cca(xb, yb, sigma=2)
            batch_cka = batch_hsic / np.sqrt(hsic_normalized_cca(xb,xb, sigma=2)* hsic_normalized_cca(yb,yb, sigma=2))

            batches_hsic.append(batch_hsic)
            batches_cka.append(batch_cka)
    print("Batches HSIC and CKA for layer with len ", len(batches_hsic), " ",len(batches_cka), flush=True)

    return torch.mean(torch.FloatTensor(batches_hsic)), torch.mean(torch.FloatTensor(batches_cka))


