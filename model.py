#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.amp as amp
from torch_harmonics import DiscreteContinuousConvS2

class INN_Encoder_FlatwPool_Multidir(nn.Module):

    def get_single_encoder_block(self, in_channels, out_channels, depth=3):

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, (3, 3)))  # first block
        for i in range(depth-1):
            layers.append(nn.Conv2d(out_channels, out_channels, (3, 3)))
            layers.append(nn.ELU())

        return layers

    def get_conv_module(self, depth=3, block_depth=3, min_channels=8, max_channels=128):

        # assert that depth is feasible
        assert self.args.n_theta >= 2**(depth-1)

        layers = []

        in_channels = self.in_size
        out_channels = min_channels
        for i in range(depth):
            layers.extend(self.get_single_encoder_block(in_channels, out_channels, depth=block_depth))
            if i < depth - 1:
                layers.append(nn.MaxPool2d((2, 2)))

            in_channels = min(out_channels, max_channels)
            out_channels = in_channels * 2

        return nn.Sequential(*layers)


    def get_fc_module(self, in_shape):
        mlp_out = nn.Sequential(
            nn.Linear(in_shape, 256, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Linear(128, self.embedding_size, bias=self.use_bias),
            nn.LeakyReLU()
        )

        return mlp_out

    def __init__(self, args):
        super(INN_Encoder_FlatwPool_Multidir, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.use_bias = args.use_bias
        self.in_size = args.in_channels_inn
        self.flat = nn.Flatten()
        self.n_theta = args.n_theta
        self.n_phi = args.n_phi
        self.depth = args.inn_depth

        self.convs_0 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the first inc direction: 0, 0
        self.convs_1 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the second inc direciton: 0, 180
        self.convs_2 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the third inc direction: 90, 90
        self.convs_3 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the foruth inc direction: 90, -90
        self.convs_4 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the fifth inc direction: 90, 0
        self.convs_5 = self.get_conv_module(args.inn_depth, args.block_depth)  # encoder for the sixth inc direction: 90, 180

        data = torch.ones((args.batch_size, self.in_size, self.n_theta, self.n_phi))
        out_shape = self.convs_0(data).view(args.batch_size, -1).shape[-1]
        print(f"out_shape in multidir encoders: {out_shape}")
        out_shape = out_shape * 6
        self.mlp_out = self.get_fc_module(out_shape)
        self.output_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=self.use_bias)

    def forward(self, fields):
        #print(fields.shape)
        #exit(0)
        x0 = self.convs_0(fields[:, 0, :, :, :])
        x0 = self.flat(x0)
        x1 = self.convs_1(fields[:, 1, :, :, :])
        x1 = self.flat(x1)
        x2 = self.convs_2(fields[:, 2, :, :, :])
        x2 = self.flat(x2)
        x3 = self.convs_3(fields[:, 3, :, :, :])
        x3 = self.flat(x3)
        x4 = self.convs_4(fields[:, 4, :, :, :])
        x4 = self.flat(x4)
        x5 = self.convs_5(fields[:, 5, :, :, :])
        x5 = self.flat(x5)

        fused_latent = torch.cat([x0, x1, x2, x3, x4, x5], dim=1)  # Concatanated
        #print(x0.shape)
        #print(fused_latent.shape)
        fused_latent = self.mlp_out(fused_latent)
        output = self.output_layer(fused_latent) #self.reparameterize(mu, std)

        return output

class AAE_Encoder(nn.Module):
    def __init__(self, args):
        super(AAE_Encoder, self).__init__()

        self.embedding_size = args.embedding_size
        self.use_bias = args.use_bias

        # Pnet Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
            nn.Conv1d(128, self.embedding_size, kernel_size=1, bias=self.use_bias),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(self.embedding_size, self.embedding_size)

        self.fc_std = nn.Linear(self.embedding_size, self.embedding_size)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.add(x1, self.conv2(x1)) # skip connection
        x3 = torch.add(x2, self.conv3(x2)) # skip connection

        x = F.adaptive_max_pool1d(x3, 1).squeeze()
        mu = self.fc_mu(x)
        std = self.fc_std(x)

        output = self.reparameterize(mu, std)

        return output, mu, std

class AAE_Generator_Micro(nn.Module):
    def __init__(self, args):
        super(AAE_Generator_Micro, self).__init__()
        self.n_points_out = args.n_points_out
        self.embedding_size = args.embedding_size
        self.use_bias = args.use_bias
        self.layers = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, args.n_points_out * 3)
        )
    def forward(self, x):
        #print("X before: ")
        #print(x.shape)
        x = self.layers(x.squeeze())
        #print("X after:")
        #print(x.shape)
        x = x.view(-1, 3, self.n_points_out)

        #print(x.shape)
        return x
