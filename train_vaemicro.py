#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import  ACS_Dataset_ordered
from data import load_airplane_multidir_ordered

from model import AAE_Encoder, AAE_Generator_Micro
import numpy as np
from torch.utils.data import DataLoader
from util import animated_PC_compare, cal_loss, IOStream, plot_3d_point_cloud, plot_PC_compare

#import sklearn.metrics as metrics
from torch.utils.data.sampler import SubsetRandomSampler

#from sklearn.model_selection import train_test_split
from scipy.stats import zscore as zsc
from itertools import chain
import wandb
from torchsummary import summary
import matplotlib.pyplot as plt
import time
from os.path import join, exists
from pytorch3d.loss import chamfer_distance


from matplotlib import animation
import tqdm
def _init_(ckpt_pth):
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
    if not os.path.exists(ckpt_pth+args.exp_name):
        os.makedirs(ckpt_pth+args.exp_name)
    if not os.path.exists(ckpt_pth+args.exp_name+'/'+'models'):
        os.makedirs(ckpt_pth+args.exp_name+'/'+'models')
    if not os.path.exists(ckpt_pth+args.exp_name+'/'+'samples'):
        os.makedirs(ckpt_pth+args.exp_name+'/'+'samples')

    os.system(f'cp train_vaemicro.py {ckpt_pth + args.exp_name}/train_vaemicro.py.backup')
    os.system(f'cp model.py {ckpt_pth + args.exp_name}/model.py.backup')
    os.system(f'cp util.py {ckpt_pth + args.exp_name}/util.py.backup')
    os.system(f'cp data.py {ckpt_pth + args.exp_name}/data.py.backup')

def train(args, io):

    torch.manual_seed(args.man_seed)
    wandb.init(project='VAE_AIRPLANE_FIXED', config=args, name=args.exp_name )
    ### X and y are numpy arrays. X is the input data, the data you feed to the first layer of the NN. y is the output, the vector you are trying to predict.
    # I use a custom function (load_airplane_multidir_ordered) below, to load chunks of data from disk and stich it together with numpy. I simplified it in this version to read a single chunk.

    # if using the same input and output PC size
    if not args.n_points_out:
        args.n_points_out = args.n_points

    labels, X, _ = load_airplane_multidir_ordered(args.data_root, args, pc_sample=args.n_points, n_theta=args.n_theta, n_phi=args.n_phi,  pc_sample_out=args.n_points_out)
    y=X


    print('Shape y/X:')
    print(y.shape)
    print(X.shape)

    # train_size and val_size determine the train and val splits, since my data is shuffled outside, I pick the first train_size for training and the val_size after those for validation
    # if you use your own data, you could shuffle it between runs
    print('Size of train set: ', args.train_size)
    print('Size of validation set: ', args.val_size)
    print('==========================================')

    # these below are passed into our dataloader, which splits the data into training and validation batches. The ranges below are the indices of the first train_size and the following val_size elements.
    train, test = range(args.train_size), range(args.train_size, args.train_size + args.val_size )


    # Get the dataset in memory
    ds = ACS_Dataset_ordered(labels, X, y)

    # I think you can use the following 2 lines as they are
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=SubsetRandomSampler(train),  drop_last=True)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=SubsetRandomSampler(test),  drop_last=False)

    # this tells torch whether to use gpu or not (try to use it, it will be terribly slow otherwise. also I really did not test it on a CPU.)
    device = torch.device("cuda" if args.cuda else "cpu")



    #Try to load models : You can implement a custom model in model.py and import it here, you'll simply add a new branch below, for the new model you implemented.
    G = AAE_Generator_Micro(args).to(device)
    E = AAE_Encoder(args).to(device)



    print(str(E))
    print(str(G))
    # print num params
    model_parameters = filter(lambda p: p.requires_grad, E.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of params: {params}')


    noise = torch.FloatTensor(args.batch_size, args.embedding_size)
    noise = noise.to(device)
    # WANDB Hook to record gradients in the model
    wandb.watch(E)
    wandb.watch(G)

    # These are the gradient descent optimizers, feel free to play with them.
    if args.use_sgd:
        print("Use SGD")
        EG_opt = optim.SGD(chain(E.parameters(), G.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        EG_opt = optim.Adam(chain(E.parameters(), G.parameters()), lr=args.lr, weight_decay=1e-4)
    # This is the learning-rate scheduler. it schedules the learning-rate values throughout the entire training epochs. This one uses cosine-annealing, you could check it online.
    # if you want to use a fixed learning rate, you could set args.eta_min equal to args.lr, this will make the minimum learning rate of the scheduler equal to the starting learning rate
    # so the scheduler will not assign new LR values.
    EG_scheduler = CosineAnnealingLR(EG_opt, args.epochs, eta_min=args.eta_min) #0.00005)

    # some training stats I want to record
    best_loss = float('inf')
    best_epoch = 0
    time_best_epoch = 0

    time_total = 0

    # save starting time
    time_epoch0 = time.time()
    normal_std = torch.tensor(args.normal_std)

    print('Starting training loop...')
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        G.train()
        E.train()


        train_pred = []
        train_true = []

        total_loss_eg = 0.0
        total_loss_shape = 0.0
        batch_count = 0
        for label, X, y in train_loader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float).squeeze()

            X = X.permute(0, 2, 1) # only for point-clouds

            # Input -> Latent Representation Encoding
            preds, mu, varlog = E(X)

            # Latent Representation -> Point Cloud Decoding
            X_recons = G(preds)
            #print(f"X shape: {X.shape}")
            #print(f"X_recons shape: {X_recons.shape}")

            loss_shape = chamfer_distance(X.permute(0, 2, 1) + 0.5, X_recons.permute(0, 2, 1) + 0.5, point_reduction='sum')[0]


            total_loss_shape += loss_shape.item() / args.n_points


            loss_kld = -0.5 * torch.mean(
                1-2.0 * torch.log(normal_std) + varlog - (mu.pow(2) + varlog.exp()) / torch.pow(normal_std, 2)
            )
            #print("KLD:" + str(loss_kld))

            loss_eg = loss_shape + args.alpha_kld * loss_kld
            EG_opt.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            total_loss_eg += loss_eg.item()
            EG_opt.step()
            batch_count += 1 #batch_size

        EG_scheduler.step()


        train_loss = total_loss_eg / batch_count #/ (args.n_points*train_count)
        train_shape_loss = total_loss_shape / batch_count #/ train_count / args.n_points
        outstr = 'Epoch %d Train EG Loss: %.7f' % (epoch, train_loss)
        outstr += '\nEpoch %d Train Shape Loss: %.7f' % (epoch, train_shape_loss)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        total_loss_shape = 0.0
        batch_count = 0

        E.eval()
        G.eval()
        test_pred = []
        test_true = []
        for label, X, y in test_loader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float).squeeze()

            X = X.permute(0, 2, 1)
            X = X.to(device)

            with torch.no_grad():
                preds, _, _ = E(X)
                X_recons = G(preds)

                loss_shape = chamfer_distance(X.permute(0, 2, 1) + 0.5, X_recons.permute(0, 2, 1) + 0.5)[0]
                total_loss_shape += loss_shape.item()


            batch_count += 1

        test_shape_loss = total_loss_shape / batch_count
        outstr = 'Epoch %d Val Shape Loss: %.6f' % (epoch, test_shape_loss)

        io.cprint(outstr)

        X_recons = X_recons.data.cpu().numpy()
        X = X.data.cpu().numpy()
        time_total = time.time() - time_epoch0

        if test_shape_loss < best_loss:
            best_loss = test_shape_loss
            best_pc = X
            best_pc_recons = X_recons
            best_epoch = epoch
            best_gen_dict = G.state_dict()
            best_enc_dict = E.state_dict()
            best_opt_dict = EG_opt.state_dict()

        if epoch % args.save_freq == 0 and epoch != 0:
            torch.save(best_gen_dict, f'{args.checkpoint_path}%s/models/G_best.t7' % args.exp_name)
            torch.save(best_enc_dict, f'{args.checkpoint_path}%s/models/E_best.t7' % args.exp_name)

            torch.save(best_opt_dict, f'{args.checkpoint_path}%s/models/EG_opt_best.t7' % args.exp_name)

            #for k in range(min(20, batch_size)):
            #  fig = plot_PC_compare(best_pc[k][0], best_pc[k][1], best_pc[k][2], best_pc_recons[k][0], best_pc_recons[k][1], best_pc_recons[k][2], epoch=epoch)
            #  fig.savefig(
            #    join(args.checkpoint_path,args.exp_name, 'samples', f'{epoch}_{k}.png'), dpi=300)
            #  plt.close(fig)

          # Save animations of test reconstructions
            for k in tqdm.tqdm(range(min(20, args.batch_size))):
              anim = animated_PC_compare(best_pc[k][0], best_pc[k][1], best_pc[k][2], best_pc_recons[k][0], best_pc_recons[k][1], best_pc_recons[k][2], title=f"{epoch}")

              anim.save(
                  join(args.checkpoint_path, args.exp_name, 'samples', f'best_{k}.gif'), writer='imagemagick'
              )

            time_best_epoch = time_total


        # I use WANDB api to log the training stats each epoch. You can use the variable names down below to store training stats and dump them periodically or at the end of the training process
        # best_loss, best_epoch, time_best_epoch etc. are still being updated, you just need to save them.
        #
        wandb.log({'loss':train_loss, 'test_shape_loss': test_shape_loss, 'epoch': epoch, 'lr': EG_scheduler.get_last_lr()[0], 'best_loss': best_loss, 'best_epoch': best_epoch, 'time_best_epoch':time_best_epoch, 'time_total': time_total})



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--alpha_kld', type=float, default=0.1)
    parser.add_argument('--save_freq', type=int, default=15)
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--data_root', type=str, default='/mnt/home/dikbayir/Research/data/npydata', help='Root path of the dataset.')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=15000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--eta_min', type=float, default=0.0000005)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=52, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--embedding_size', type=int, default=64, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_points', type=int, default=2048, help='Num points in the in point-cloud.')
    parser.add_argument('--n_points_out', type=int, default=None, help='Num points in the out point-cloud.')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--reconstruction_loss', type=str, default='chamfer')

    parser.add_argument('--norm', type=str, default='none', choices=['none', 'minmax', 'std']) # input normalization type, for no normalization use 'none'
    parser.add_argument('--train_size', type=int, default=200) # the size of the training set
    parser.add_argument('--val_size', type=int, default=20) # the size of the validation set


    parser.add_argument('--reduction', type=str, default='sum')
    parser.add_argument('--reconstruction_coef', type=float, default=1.0)
    parser.add_argument('--normal_mu', type=float, default=0.0)
    parser.add_argument('--normal_std', type=float, default=0.2)
    parser.add_argument('--use_bias', type=bool, default=True)
    parser.add_argument('--n_theta', type=int, default=64)
    parser.add_argument('--n_phi', type=int, default=128)

    parser.add_argument('--man_seed', type=int, default=42)
    args = parser.parse_args()

    exp_name_str = [args.exp_name, f'kld:{args.alpha_kld}', f'lr:{args.lr}', f'std:{args.normal_std}', f'eta_min:{args.eta_min}', f'in_points:{args.n_points}', f'out_points:{args.n_points_out}']
    args.exp_name = '_'.join(exp_name_str)

    _init_(args.checkpoint_path)

    io = IOStream(args.checkpoint_path + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
