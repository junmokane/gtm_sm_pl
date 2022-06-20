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

# python -m smp_v0_train vae.model_path=./result/vae/\[2022-05-11\ 21\:29\:33\]/checkpoints/model_epoch_800.ckpt
# tensorboard --logdir ./result/gtm_sm/\[2022-06-19\ 18\:18\:30\]tm_sm/version_0/ --samples_per_plugin "images=512"

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
    # make result folder
    args.result_dir = os.path.join(args.result_dir, args.exp_name, f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]')
    args.ckpt_dir = os.path.join(args.result_dir, 'checkpoints')
    
    # data module
    dm = gtm_sm_data_module(args)
    dm.setup()
    
    train_dataloader = dm.train_dataloader()
    sample = next(iter(train_dataloader))
    for i in range(len(sample)):
        print(sample[i].size(), sample[i].max(), sample[i].min())
    
    # gtm_sm model
    model = gtm_sm(args)
    # model.inference(batch=sample[0])
    # exit()

    logger = TensorBoardLogger(
        save_dir=args.result_dir,
        name=args.exp_name
    )

    trainer = pl.Trainer(
        logger=logger,
        # gpus=args.train.gpus,
        gpus=list(range(len(args.train.gpus))),
        progress_bar_refresh_rate=args.log.print_step_freq,
        accelerator=args.accelerator,
        check_val_every_n_epoch=args.train.check_val_every_n_epoch,
        max_epochs=args.train.max_epochs,
        gradient_clip_val=args.train.gradient_clip_val,
        limit_val_batches=args.train.limit_val_batches,
        checkpoint_callback=False,
        # default_root_dir=ckpt_dir,
        # callbacks=[MyTrainLoopCallback()],
        benchmark=False,
        deterministic=True,
        # truncated_bptt_steps=args.train.truncated_bptt_steps,
        # profiler="pytorch",
        plugins=[DDPPlugin(find_unused_parameters=True)],
        # plugins='deepspeed',
        # profiler=True,
    )

    trainer.fit(
        model,
        datamodule=dm
    )

if __name__ == '__main__':
    main()