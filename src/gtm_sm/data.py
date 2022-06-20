import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.datasets as dset
import cv2
from PIL import Image
import matplotlib.pyplot as plt

'''
data module for gtm_sm
'''
class gtm_sm_data_module(pl.LightningDataModule):
    def __init__(self, args):
        super(gtm_sm_data_module, self).__init__()
        self.args = args
        
    def setup(self, stage = None):
        data_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),])
        
        self.train_dataset = dset.ImageFolder(root=self.args.train_dir, transform=data_transform)
        self.valid_dataset = dset.ImageFolder(root=self.args.val_dir, transform=data_transform)
        self.test_dataset = dset.ImageFolder(root=self.args.test_dir, transform=data_transform)
        
        if 'ddp' in self.args.accelerator:
            self.batch_size = self.args.train.batch_size // len(self.args.train.gpus)
        else:
            self.batch_size = self.args.train.batch_size        
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train.batch_size, shuffle=True,
            num_workers=self.args.train.num_workers, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.args.train.batch_size, shuffle=False,
            num_workers=self.args.train.num_workers, pin_memory=True, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.args.train.batch_size, shuffle=False,
            num_workers=self.args.train.num_workers, pin_memory=True, drop_last=True
        )
