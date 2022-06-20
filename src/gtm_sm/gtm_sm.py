import os
import numpy as np
import torch
import random
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.distributions import OneHotCategorical
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning import Callback
from einops import rearrange, repeat, reduce
import torch.distributed as dist
import pyflann
from PIL import Image

from typing import List, Dict, Tuple, Any, Optional, Callable, Union, Sequence
from src.utils import Preprocess_img, Flatten, Exponent, Unflatten, Deprocess_img
from src.roam import random_walk_wo_wall

class gtm_sm(pl.LightningModule):

    def __init__(self, args):
        super(gtm_sm, self).__init__()

        self.save_hyperparameters(args)
        num_digit = int(np.log10(self.hparams.train.max_epochs)) + 1
        self.save_name_template = f'model_epoch_{{0:0{num_digit}d}}.ckpt'
        
        self.flanns = pyflann.FLANN()

        self.enc_zt = nn.Sequential(
            Preprocess_img(),
            nn.Conv2d(3, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.01),
            Flatten()
        )

        self.enc_zt_mean = nn.Sequential(
            nn.Linear(64, self.hparams.gtm_sm.z_dim))

        self.enc_zt_std = nn.Sequential(
            nn.Linear(64, self.hparams.gtm_sm.z_dim),
            Exponent())

        # for st
        self.enc_st_matrix = nn.Sequential(
            nn.Linear(self.hparams.gtm_sm.a_dim, self.hparams.gtm_sm.s_dim, bias=False))

        self.enc_st_sigmoid = nn.Sequential(
            nn.Linear(self.hparams.gtm_sm.s_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.hparams.gtm_sm.z_dim, 64),
            nn.ReLU(),
            Unflatten(-1, 16, 2, 2),
            nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
            nn.Tanh(),
            Deprocess_img())

    @rank_zero_only
    def breakpoint_here(self):
        breakpoint()
        return

    def inference(self, batch):
        '''
        batch: b,c,h,w (16,3,32,32)
        '''
        # print(batch.size(), batch.max(), batch.min())
        x, _ = batch
        action_one_hot_value, position, action_selection = random_walk_wo_wall(self)
        
        st_observation_list = []
        st_prediction_list = []
        zt_mean_observation_list = []
        zt_std_observation_list = []
        zt_mean_prediction_list = []
        zt_std_prediction_list = []
        xt_prediction_list = []

        kld_loss = 0
        nll_loss = 0
        
        # observation phase: construct st
        for t in range(self.hparams.gtm_sm.observe_dim):
            if t == 0:
                st_observation_t = torch.zeros(self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim, device=self.device)#torch.rand(self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim, self.device=self.device) - 1
            else:
                replacement = self.enc_st_matrix(action_one_hot_value[:, :, t - 1])  # [16,2]
                st_observation_t = st_observation_list[t - 1] + replacement + \
                                    torch.randn((self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim), device=self.device) * self.hparams.gtm_sm.r_std
            st_observation_list.append(st_observation_t)
        st_observation_tensor = torch.cat(st_observation_list, 0).view(self.hparams.gtm_sm.observe_dim, self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim)

        # prediction phase: construct st
        for t in range(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim):
            if t == 0:
                replacement = self.enc_st_matrix(action_one_hot_value[:, :, t + self.hparams.gtm_sm.observe_dim - 1])
                st_prediction_t = st_observation_list[-1] + replacement + \
                                    torch.randn((self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim), device=self.device) * self.hparams.gtm_sm.r_std
            else:
                replacement = self.enc_st_matrix(action_one_hot_value[:, :, t + self.hparams.gtm_sm.observe_dim - 1])
                st_prediction_t = st_prediction_list[t - 1] + replacement + \
                                    torch.randn((self.hparams.train.batch_size, self.hparams.gtm_sm.s_dim), device=self.device) * self.hparams.gtm_sm.r_std
            st_prediction_list.append(st_prediction_t)
        st_prediction_tensor = torch.cat(st_prediction_list, 0).view(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim, self.hparams.train.batch_size,
                                                                     self.hparams.gtm_sm.s_dim)
        # observation phase: construct zt from xt
        for t in range(self.hparams.gtm_sm.observe_dim):
            index_mask = torch.zeros((self.hparams.train.batch_size, 3, 32, 32), device=self.device)
            for index_sample in range(self.hparams.train.batch_size):
                position_h_t = position[index_sample, 0, t]
                position_w_t = position[index_sample, 1, t]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                3 * position_w_t:3 * position_w_t + 8] = 1
            index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_observation_t = self.enc_zt(x_feed)
            zt_mean_observation_t = self.enc_zt_mean(zt_observation_t)
            zt_std_observation_t = self.enc_zt_std(zt_observation_t)
            zt_mean_observation_list.append(zt_mean_observation_t)
            zt_std_observation_list.append(zt_std_observation_t)
        zt_mean_observation_tensor = torch.cat(zt_mean_observation_list, 0).view(self.hparams.gtm_sm.observe_dim, self.hparams.train.batch_size,
                                                                                 self.hparams.gtm_sm.z_dim)
        zt_std_observation_tensor = torch.cat(zt_std_observation_list, 0).view(self.hparams.gtm_sm.observe_dim, self.hparams.train.batch_size,
                                                                               self.hparams.gtm_sm.z_dim)
        # prediction phase: construct zt from xt
        for t in range(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim):
            index_mask = torch.zeros((self.hparams.train.batch_size, 3, 32, 32), device=self.device)
            for index_sample in range(self.hparams.train.batch_size):
                position_h_t = position[index_sample, 0, t + self.hparams.gtm_sm.observe_dim]
                position_w_t = position[index_sample, 1, t + self.hparams.gtm_sm.observe_dim]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                3 * position_w_t:3 * position_w_t + 8] = 1
            index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_prediction_t = self.enc_zt(x_feed)
            zt_mean_prediction_t = self.enc_zt_mean(zt_prediction_t)
            zt_std_prediction_t = self.enc_zt_std(zt_prediction_t)
            zt_mean_prediction_list.append(zt_mean_prediction_t)
            zt_std_prediction_list.append(zt_std_prediction_t)
        zt_mean_prediction_tensor = torch.cat(zt_mean_prediction_list, 0).view(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim,
                                                                                self.hparams.train.batch_size, self.hparams.gtm_sm.z_dim)
        zt_std_prediction_tensor = torch.cat(zt_std_prediction_list, 0).view(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim,
                                                                                self.hparams.train.batch_size, self.hparams.gtm_sm.z_dim)

        # reparameterized_sample to calculate the reconstruct error
        for t in range(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim):
            zt_prediction_sample = self._reparameterized_sample(zt_mean_prediction_list[t],
                                                                zt_std_prediction_list[t])
            index_mask = torch.zeros((self.hparams.train.batch_size, 3, 32, 32), device=self.device)
            for index_sample in range(self.hparams.train.batch_size):
                position_h_t = position[index_sample, 0, t + self.hparams.gtm_sm.observe_dim]
                position_w_t = position[index_sample, 1, t + self.hparams.gtm_sm.observe_dim]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                3 * position_w_t:3 * position_w_t + 8] = 1
            index_mask_bool = index_mask.ge(0.5)
            x_ground_true_t = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            x_resconstruct_t = self.dec(zt_prediction_sample)
            nll_loss += self._nll_gauss(x_resconstruct_t, x_ground_true_t)
            xt_prediction_list.append(x_resconstruct_t)

        # construct kd tree
        st_observation_memory = st_observation_tensor.cpu().detach().numpy()
        st_prediction_memory = st_prediction_tensor.cpu().detach().numpy()

        results = []
        for index_sample in range(self.hparams.train.batch_size):
            param = self.flanns.build_index(st_observation_memory[:, index_sample, :], algorithm='kdtree',
                                            trees=4)
            result, _ = self.flanns.nn_index(st_prediction_memory[:, index_sample, :],
                                             self.hparams.gtm_sm.knn, checks=param["checks"])
            results.append(result)
            
        # calculate the kld
        for index_sample in range(self.hparams.train.batch_size):
            knn_index = results[index_sample]
            knn_index_vec = np.reshape(knn_index, (self.hparams.gtm_sm.knn * (self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim)))
            knn_st_memory = (st_observation_tensor[knn_index_vec, index_sample]).reshape((self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim), \
                                                                                self.hparams.gtm_sm.knn, -1)
            dk2 = ((knn_st_memory.transpose(0, 1) - st_prediction_tensor[:, index_sample, :]) ** 2).sum(2).transpose(0, 1)
            wk = 1 / (dk2 + self.hparams.gtm_sm.delta)
            normalized_wk = (wk.t() / torch.sum(wk, 1)).t()
            log_normalized_wk = torch.log(normalized_wk)
            zt_sampling = self._reparameterized_sample_cluster(zt_mean_prediction_tensor[:, index_sample],
                                                                zt_std_prediction_tensor[:, index_sample])
                                                                # [1000, 32, 2]
            log_q_phi = - 0.5 * self.hparams.gtm_sm.z_dim * torch.log(torch.tensor(2 * 3.1415926535, device = self.device)) - \
                0.5 * self.hparams.gtm_sm.z_dim - torch.log(zt_std_prediction_tensor[:, index_sample]).sum(1)
            zt_mean_knn_tensor = zt_mean_observation_tensor[knn_index_vec, index_sample].reshape(
                (self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim), self.hparams.gtm_sm.knn, -1)
            zt_std_knn_tensor = zt_std_observation_tensor[knn_index_vec, index_sample].reshape(
                (self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim), self.hparams.gtm_sm.knn, -1)

            log_p_theta_element = self._log_gaussian_element_pdf(zt_sampling, zt_mean_knn_tensor, zt_std_knn_tensor) + \
                                    log_normalized_wk
            (log_p_theta_element_max, _) = torch.max(log_p_theta_element, 2)
            log_p_theta_element_nimus_max = (log_p_theta_element.transpose(1, 2).transpose(0, 1) - log_p_theta_element_max)
            p_theta_nimus_max = torch.exp(log_p_theta_element_nimus_max).sum(0)
            kld_loss += torch.mean(log_q_phi - torch.mean(log_p_theta_element_max + torch.log(p_theta_nimus_max), 0))
        
        return kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position
        #return rnn_hid_list, cross_entropy_all, vis_entropy_all, log

    def forward(self, batch):

        return self.inference(batch)

    def training_step(self, batch, batch_idx):
        
        self.hparams.global_step = self.global_step

        self.hparams.log.phase_log = self.global_step % self.hparams.log.print_step_freq == 0

        kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = self(batch)
        
        total_loss = (nll_loss + kld_loss) / self.hparams.train.batch_size
        
        out_log = {
            'loss': total_loss,
            'log': [batch[0], st_observation_list, st_prediction_list, xt_prediction_list, position],
        }

        if self.hparams.log.phase_log:
            self.log_tb(out_log)

        return out_log

    def training_epoch_end(self, outputs):

        if self.current_epoch % self.hparams.log.save_epoch_freq == 0:
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.ckpt_dir,
                    self.save_name_template.format(self.current_epoch)
                )
            )
        return

    def on_train_start(self) -> None:
        self.trainer.save_checkpoint(
            os.path.join(
                self.hparams.ckpt_dir,
                self.save_name_template.format(-1)
            )
        )
        return

    def configure_optimizers(self):
        
        param = [
            {
                'params': [v for k, v in self.named_parameters()],
                'lr': self.hparams.train.lr
            }
        ]

        optimizer = torch.optim.Adam(param)
        return optimizer

    @rank_zero_only
    def log_tb(self, log) -> None:
        self.logger.experiment.add_scalar(f'train/total_loss', log['loss'], self.global_step)
        x, st_observation_list, st_prediction_list, xt_prediction_list, position = log['log']
        # the shape should be c,h,w
        sample_traj_vis = self.show_experiment_information(x, st_observation_list, st_prediction_list, xt_prediction_list, position) 
        self.logger.experiment.add_image(f'train/sample_traj_vis', sample_traj_vis, self.global_step)
        return

    def _log_gaussian_element_pdf(self, zt, zt_mean, zt_std):
        constant_value = torch.tensor(2 * 3.1415926535, device=self.device)
        zt_repeat = zt.unsqueeze(2).repeat(1, 1, self.hparams.gtm_sm.knn, 1)
        log_exp_term = - torch.sum((((zt_repeat - zt_mean) ** 2) / (zt_std ** 2) / 2.0), 3)
        log_other_term = - (self.hparams.gtm_sm.z_dim / 2.0) * torch.log(constant_value) - torch.sum(torch.log(zt_std), 2)
        return log_exp_term + log_other_term

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn_like(std, device=self.device)
        return eps.mul(std).add(mean)

    def _reparameterized_sample_cluster(self, mean, std):
        """using std to sample"""
        eps = torch.randn((self.hparams.gtm_sm.kl_samples, self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim, self.hparams.gtm_sm.z_dim), device=self.device)
        return eps.mul(std).add(mean)

    def _nll_gauss(self, x, mean):
        # n, _ = x.size()
        return torch.sum((x - mean) ** 2)
    
    def show_experiment_information(self, x, st_observation_list, st_prediction_list, xt_prediction_list, position):

        sample_id = np.random.randint(0, self.hparams.train.batch_size, size=(1))
        sample_imgs = x[sample_id]

        st_observation_sample = np.zeros((self.hparams.gtm_sm.observe_dim, self.hparams.gtm_sm.s_dim))
        for t in range(self.hparams.gtm_sm.observe_dim):
            st_observation_sample[t] = st_observation_list[t][sample_id].cpu().detach().numpy()

        st_prediction_sample = np.zeros((self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim, self.hparams.gtm_sm.s_dim))
        for t in range(self.hparams.gtm_sm.total_dim - self.hparams.gtm_sm.observe_dim):
            st_prediction_sample[t] = st_prediction_list[t][sample_id].cpu().detach().numpy()

        st_2_max = np.maximum(np.max(st_observation_sample[:, 0]), np.max(st_prediction_sample[:, 0]))
        st_2_min = np.minimum(np.min(st_observation_sample[:, 0]), np.min(st_prediction_sample[:, 0]))
        st_1_max = np.maximum(np.max(st_observation_sample[:, 1]), np.max(st_prediction_sample[:, 1]))
        st_1_min = np.minimum(np.min(st_observation_sample[:, 1]), np.min(st_prediction_sample[:, 1]))
        axis_st_1_max = st_1_max + (st_1_max - st_1_min) / 10.0
        axis_st_1_min = st_1_min - (st_1_max - st_1_min) / 10.0
        axis_st_2_max = st_2_max + (st_2_max - st_2_min) / 10.0
        axis_st_2_min = st_2_min - (st_2_max - st_2_min) / 10.0

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        fig.clf()
        gs = fig.add_gridspec(2, 2)

        # plot position trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Observation (256 steps)')
        ax1.set_aspect('equal')
        plt.axis([-1, 9, -1, 9])
        plt.gca().invert_yaxis()
        plt.plot(position[sample_id, 1, :self.hparams.gtm_sm.observe_dim].T, position[sample_id, 0, :self.hparams.gtm_sm.observe_dim].T, color='k',
                linestyle='solid', marker='o')
        plt.plot(position[sample_id, 1, self.hparams.gtm_sm.observe_dim-1], position[sample_id, 0, self.hparams.gtm_sm.observe_dim-1], 'bs')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Prediction (256 steps)')
        ax2.set_aspect('equal')
        plt.axis([-1, 9, -1, 9])
        plt.gca().invert_yaxis()
        plt.plot(position[sample_id, 1, self.hparams.gtm_sm.observe_dim:].T, position[sample_id, 0, self.hparams.gtm_sm.observe_dim:].T, color='k',
                linestyle='solid', marker='o')
        plt.plot(position[sample_id, 1, -1], position[sample_id, 0, -1], 'bs')

        # plot inferred states
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_xlabel('$s_1$')
        ax3.set_ylabel('$s_2$')
        ax3.set_title('Inferred states')
        ax3.set_aspect('equal')
        plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
        plt.gca().invert_yaxis()
        plt.plot(st_observation_sample[0:, 1].T, st_observation_sample[0:, 0].T, color='k',
                linestyle='solid', marker='o')
        plt.plot(st_observation_sample[-1, 1], st_observation_sample[-1, 0], 'bs')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_xlabel('$s_1$')
        ax4.set_ylabel('$s_2$')
        ax4.set_title('Inferred states')
        ax4.set_aspect('equal')
        plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
        plt.gca().invert_yaxis()
        plt.plot(st_prediction_sample[0:, 1].T, st_prediction_sample[0:, 0].T, color='k',
                linestyle='solid', marker='o')
        plt.plot(st_prediction_sample[-1, 1], st_prediction_sample[-1, 0], 'bs')
        
        plt.close()
        
        fig.canvas.draw()
        fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig_np = fig_np.transpose(2, 0, 1)
        return fig_np