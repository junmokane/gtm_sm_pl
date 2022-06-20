import os
import time
import torch
import math
import yaml
import shutil

import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
from pytorch_lightning.plugins import DDPPlugin

from omegaconf import OmegaConf

from src.gtm_sm.data import gtm_sm_data_module
from src.gtm_sm.gtm_sm import gtm_sm
from src.show_results import show_experiment_information

# python -m gtm_sm_test ckpt_dir=result/gtm_sm/\[2022-06-17\ 12\:56\:06\]/checkpoints/model_epoch_090.ckpt 

def main():
    args_base = OmegaConf.load('src/config/gtm_sm/default_config.yaml')
    args_cli = OmegaConf.from_cli()
    print(f'command line input: {args_cli}')
    args = OmegaConf.merge(args_base, args_cli)
    print(f'argument list: {args}')
    # env setting
    os.environ['PL_GLOBAL_SEED'] = f'{args.train.seed}'
    pl.trainer.seed_everything(args.train.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train.gpus)
    
    # data module
    dm = gtm_sm_data_module(args)
    dm.setup()
    val_dataloader = dm.val_dataloader()
    batch = next(iter(val_dataloader))
    for i in range(len(batch)):
        print(batch[i].size(), batch[i].max(), batch[i].min())
    
    # gtm_sm model
    model = gtm_sm(args)
    if args.ckpt_dir:
        model = model.load_from_checkpoint(args.ckpt_dir)
        model.hparams.gtm_sm.total_dim = 512
        model.eval()
        model.freeze()
    else:
        print('model is not loaded correctly.')
        exit()

    # evaluate
    kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = model.inference(batch)
    exp = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]'
    fname = f"./test/{exp}/"
    os.makedirs(fname, exist_ok=True)
    show_experiment_information(model, batch[0], st_observation_list, st_prediction_list, xt_prediction_list, position, fname)


if __name__ == '__main__':
    main()