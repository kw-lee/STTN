# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:46:02 2021

@author: wzhangcd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import numpy as np
import pandas as pd

import sys
sys.path.append('./lib')
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
import torch.utils.data as data

from time import time
import shutil
import argparse
# import configparser
import os

from ST_Transformer_new import STTransformer # STTN model with linear layer to get positional embedding
from ST_Transformer_new_sinembedding import STTransformer_sinembedding #STTN model with sin()/cos() to get positional embedding, the same as "Attention is all your need"

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # save **kwargs to self.hparams
        self.save_hyperparameters()

        adj_mx = pd.read_csv(self.hparams.adj_mat_path, header = None)
        adj_mx = np.array(adj_mx)
        self.A = torch.Tensor(adj_mx)

        self.net = STTransformer(
            adj=self.A,
            in_channels=self.hparams.in_channels,
            embed_size=self.hparams.embed_size,
            time_num=self.hparams.time_num, 
            num_layers=self.hparams.num_layers, 
            T_dim=self.hparams.T_dim, 
            output_T_dim=self.hparams.output_T_dim, 
            heads=self.hparams.heads,
            cheb_K=self.hparams.cheb_K,
            forward_expansion=self.hparams.forward_expansion,
            dropout=self.hparams.dropout
        )  

    def forward(self, *args):
        return self.net(*args)

    def MSE_loss(self, *args):
        return F.mse_loss(*args)

    def step(self, batch, batch_idx, state='train'):
        encoder_inputs, labels = batch
        outputs = self(encoder_inputs.permute(0, 2, 1, 3))   
        loss = self.MSE_loss(outputs, labels)  
        # if state == 'train': 
        #     self.log('train_loss_step', float(loss.cpu().detach()), on_step=True, prog_bar=True)
        return {'loss': loss}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, state='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, state='validation')

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        self.log(state+'_loss', float(loss), on_step=False, on_epoch=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')
    
    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='validation')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = optim.AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }           

    def dataloader(self, type='train', shuffle=False):
        """ 
        type: 'train', 'val', test'
        """
        file_data = np.load(self.hparams.filename)

        data_x = file_data[f'{type}_x']  # (10181, 307, 3, 12)
        data_x = data_x[:, :, 0:1, :]
        data_target = file_data[f'{type}_target']  # (10181, 307, 12)

        # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
        # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

        data_x_tensor = torch.from_numpy(data_x).type(torch.FloatTensor)
        # (B, N, F, T)
        data_target_tensor = torch.from_numpy(data_target).type(torch.FloatTensor)
        # (B, N, T)

        data_dataset = data.TensorDataset(data_x_tensor, data_target_tensor)
        data_loader = data.DataLoader(data_dataset, batch_size=self.hparams.batch_size, shuffle=shuffle)
        return data_loader
        
    def train_dataloader(self):
        return self.dataloader(type='train', shuffle=True)

    def val_dataloader(self):
        return self.dataloader(type='val', shuffle=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training Parameters')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=0.1)

    argp = parser.parse_args()

    args = {
        'random_seed': 24, 

        ## Path for saving network parameters
        'params_path': './Experiment/PEMS25_embed_size64', 

        ## Data generated by prepareData.py
        'filename': './PEMSD7/V_25_r1_d0_w0_astcgn.npz',

        ## The same setting as prepareData.py
        'num_of_hours': 1, 
        'num_of_days': 0,
        'num_of_weeks': 0 ,

        ## Adjacency Matrix Import
        'adj_mat_path': './PEMSD7/W_25.csv',

        ## Training
        'batch_size': argp.batch_size,
        'lr': argp.lr,
        'epochs': argp.epochs,
        'optimizer': 'AdamW',  # AdamW vs AdamP
        'lr_scheduler': 'cos',  # ExponentialLR vs CosineAnnealingWarmRestarts

        ## Model
        'in_channels': 1, # Channels of input
        'embed_size': 64, # Dimension of hidden embedding features
        'time_num': 288, 
        'num_layers': 3, # Number of ST Block
        'T_dim': 12, # Input length, should be the same as prepareData.py
        'output_T_dim': 12, # Output Expected length
        'heads': 2, # Number of Heads in MultiHeadAttention
        'cheb_K': 2, # Order for Chebyshev Polynomials (Eq 2)
        'forward_expansion': 4, # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
        'dropout': 0
    }
    
    checkpoint_callback = ModelCheckpoint(
        filename='epoch{epoch}-validation_loss{validation_loss:.4f}',
        monitor='validation_loss',
        save_top_k=5,
        mode='min',
        auto_insert_metric_name=False,
    )

    ### Training Process
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])
    
    model = Model(**args)    

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['epochs'],
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=1
    )   
  
    trainer.fit(model)
    