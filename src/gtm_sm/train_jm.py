import wandb
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multiprocessing import Process

from utils.torch_utils import initNetParams, ChunkSampler, show_images, device_agnostic_selection, show_heatmap
from model import GTM_SM
from config import *
from show_results import show_experiment_information
from roam import sample_position

def train(epoch, model, optimizer, loader_train):
    model.train()
    BCE_loss = nn.BCELoss()
    train_loss = 0
    bce_loss = 0

    for batch_idx, (data, _) in enumerate(loader_train):
        if epoch == 1 and batch_idx == 0:
            optimizer.__init__(
                [{'params': model.enc_zt.parameters()},
                    {'params': model.enc_zt_mean.parameters()},
                    {'params': model.enc_zt_std.parameters()},
                    {'params': model.enc_st_matrix.parameters()},
                    {'params': model.dec.parameters()},
                    {'params': model.enc_st_sigmoid.parameters(), 'lr': 1e-2}],
                lr=1e-3)

        # transforming data
        training_data = data.to(device=device)  # [16,3,32,32]

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, matrix_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = model.forward(
            training_data)

        train_loss += (nll_loss + kld_loss).item()
        loss_to_optimize = (nll_loss + kld_loss) / args.batch_size
        loss_to_optimize.backward()
        
        # grad norm clipping, only in pytorch version >= 1.10
        #nn.utils.clip_grad_norm_(GTM_SM_model.parameters(), args.gradient_clip)

        optimizer.step()

        # printing
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f} \t MATRIX Lose: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_train.dataset),
                        100. * batch_idx * len(data) / len(loader_train.dataset),
                        kld_loss.item() / len(data),
                        nll_loss.item() / len(data),
                        matrix_loss.item()))
            # log wandb
            wandb.log({"epoch": epoch,
                        "train_loss": (kld_loss.item() + nll_loss.item()) / len(data),
                        "kld_loss": kld_loss.item() / len(data),
                        "nll_loss":  nll_loss.item() / len(data),
                        "matrix_loss":  matrix_loss.item(),
                        "bce_loss": bce_loss,})

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader_train.dataset)))

def test(epoch, model, loader_val):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader_val):
            data = data.to(device=device)
            kld_loss, nll_loss, matrix_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = model.forward(
                data)
            test_loss += nll_loss

    test_loss /= len(loader_val.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
