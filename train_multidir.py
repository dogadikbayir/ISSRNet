#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import copy
import argparse
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import ACS_Dataset_ordered
from data import load_airplane_multidir_ordered
from model import AAE_Encoder, AAE_Generator_Micro, INN_Encoder_FlatwPool_Multidir, INN_Encoder_Conv_FlatwPool_Large, PointNetEnNoBN, INN_Encoder_SH, PointNetPubl, AAE_Generator
import numpy as np
from torch.utils.data import DataLoader
from util import animated_PC_compare, cal_loss, IOStream, plot_3d_point_cloud, plot_PC_compare

import sklearn.metrics as metrics
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
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

torch.autograd.set_detect_anomaly(True)
def _init_(ckpt_pth):
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
    if not os.path.exists(ckpt_pth+args.exp_name):
        os.makedirs(ckpt_pth+args.exp_name)
    if not os.path.exists(ckpt_pth + args.exp_name+'/'+'models'):
        os.makedirs(ckpt_pth+args.exp_name+'/'+'models')
    if not os.path.exists(ckpt_pth+args.exp_name+'/'+'samples'):
        os.makedirs(ckpt_pth+args.exp_name+'/'+'samples')

    os.system(f'cp train_conv_airplane.py {ckpt_pth + args.exp_name}/train_conv_airplane.py.backup')
    os.system(f'cp model.py {ckpt_pth + args.exp_name}/model.py.backup')
    os.system(f'cp util.py {ckpt_pth +  args.exp_name}/util.py.backup')
    os.system(f'cp data.py {ckpt_pth + args.exp_name}/data.py.backup')

def normalize_minmax(args, y):

    # Real
    y_train = y[:args.train_size, ..., 0]
    y_val = y[args.train_size:args.train_size + args.val_size, ..., 0]

    min_train, max_train = np.min(y_train), np.max(y_train)

    y_train = 2 * ((y_train - min_train) / (max_train - min_train)) - 1
    y_val = 2 * ((y_val - min_train) / (max_train - min_train)) - 1

    y[:args.train_size, ..., 0] = y_train
    y[args.train_size: args.train_size + args.val_size, ..., 0] = y_val

    # Imag
    y_train = y[:args.train_size, ..., 1]
    y_val = y[args.train_size:args.train_size + args.val_size, ..., 1]

    min_train, max_train = np.min(y_train), np.max(y_train)

    y_train = 2 * ((y_train - min_train) / (max_train - min_train)) - 1
    y_val = 2 * ((y_val - min_train) / (max_train - min_train)) - 1

    y[:args.train_size, ..., 1] = y_train
    y[args.train_size: args.train_size + args.val_size, ..., 1] = y_val


    return y


def debug_model(args):

    device = torch.device("cuda" if args.cuda else "cpu")

    INN_E = INN_Encoder_FlatwPool_Multidir(args).to(device)

    args.embedding_size = 64
    args.use_bias = True
    args.in_channels_inn = 1
    args.n_theta = 51
    args.n_phi = 101
    data_batch = torch.from_numpy(np.random.randn(32, args.n_theta, args.n_phi, 1).astype(np.float32)).to(device)
    data_batch = data_batch.permute(0, 3, 1, 2)

    INN_E(data_batch, data_batch, data_batch, data_batch, data_batch, data_batch)
    exit()

def train(args):

    torch.manual_seed(args.man_seed)
    wandb.init(project=args.project_name, config=args, name=args.exp_name)
    ### X and y are numpy arrays. X is the input data, the data you feed to the first layer of the NN. y is the output, the vector you are trying to predict.
    # I use a custom function (load_ACS) below, to load chunks of data from disk and stich it together with numpy. I simplified it in this version to read a single chunk. you can replace X and y with any array, as long as their dimensions aggree with the
    # first and last layers of the NN you want to employ.
    args.n_points_out = args.n_points

    # Original
    #X, y = load_airplane_data(args.data_root, gts_mode=args.gts_mode, lmax=args.lmax, class_obj=args.class_obj, n_sample=args.n_points, noise=args.noise)
    # EM Data
    #X, y = load_airplane_EM_data(args.data_root)
    labels, pcs, fields = load_airplane_multidir_ordered(args.data_root, args,  isMC=False, n_theta=args.n_theta, n_phi=args.n_phi, pc_sample=args.n_points, pc_sample_out=args.n_points, snr=-11)  # no noise during training
    #print(pcs.shape)
    #exit(0)
    # train_size and val_size determine the train and val splits, since my data is shuffled outside, I pick the first train_size for training and the val_size after those for validation
    # if you use your own data, you could shuffle it between runs
    print('Size of train set: ', args.train_size)
    print('Size of validation set: ', args.val_size)
    print('==========================================')

    # these below are passed into our dataloader, which splits the data into training and validation batches. The ranges below are the indices of the first train_size and the following val_size elements.
    train, test = range(args.train_size), range(args.train_size, args.train_size + args.val_size )

    if args.ffd_norm != None and args.ffd_norm == 'minmax' and args.gts_mode == 'hgts':
        print('Normalizing FFD into -1, 1 ....')
        y = normalize_minmax(args, y)
    elif args.ffd_norm != None:
        print('Not implemented this normalization type...')
        exit(0)

    # Get the dataset in memory
    ds = ACS_Dataset_ordered(labels, fields, pcs)

    # I think you can use the following 2 lines as they are
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=SubsetRandomSampler(train),  drop_last=True)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=SubsetRandomSampler(test),  drop_last=False)

    # this tells torch whether to use gpu or not (try to use it, it will be terribly slow otherwise. also I really did not test it on a CPU.)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models : You can implement a custom model in model.py and import it here, you'll simply add a new branch below, for the new model you implemented.
    if args.large:
        G = AAE_Generator(args).to(device)
    else:
        G = AAE_Generator_Micro(args).to(device)

    # Load the weights for the pretrained (fixed) modules
    G.load_state_dict(torch.load(args.G_path))


    INN_E = INN_Encoder_FlatwPool_Multidir(args).to(device)

    # print model
    print(str(INN_E))


    # WANDB Hook to record gradients in the model
    wandb.watch(INN_E)

    # These are the gradient descent optimizers, feel free to play with them.
    if args.use_sgd:
        print("Use SGD")
        E_opt = optim.SGD(INN_E.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        E_opt = optim.Adam(INN_E.parameters(), lr=args.lr, weight_decay=1e-4)
    # This is the learning-rate scheduler. it schedules the learning-rate values throughout the entire training epochs. This one uses cosine-annealing, you could check it online.
    # if you want to use a fixed learning rate, you could set args.eta_min equal to args.lr, this will make the minimum learning rate of the scheduler equal to the starting learning rate
    # so the scheduler will not assign new LR values.
    E_scheduler = CosineAnnealingLR(E_opt, args.epochs, eta_min=0.00005)

    # Loss function for the predicted ffld coefficients
    if args.loss == 'mse':
        latent_loss_f = F.mse_loss
    else:
        latent_loss_f = F.mae_loss

    # some training stats I want to record
    best_loss = float('inf')
    best_epoch = 0
    time_best_epoch = 0

    time_total = 0

    # save starting time
    time_epoch0 = time.time()
    normal_std = torch.tensor(args.normal_std)
    print('Starting training loop...')

    # Print number of parameters in the INN model
    model_parameters = filter(lambda p: p.requires_grad, INN_E.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    G.eval()
    print(f'Number of params: {params}')


    for epoch in range(args.epochs):
        ####################
        # Train
        ####################

        train_loss = 0.0
        train_count = 0.0

        #E.eval()

        INN_E.train()

        train_pred = []
        train_true = []

        total_recons_loss = 0.0
        total_loss_inn_enc = 0.0
        total_shape_loss = 0.0
        batch_count = 0
        for label, fields, pc in train_loader:


            pc, fields = pc.to(device, dtype=torch.float), fields.to(device, dtype=torch.float)

            pc = pc.permute(0, 2, 1) # only for point-clouds

            E_opt.zero_grad()
            batch_size = pc.size()[0]

            #print(f"field_shape: {fields.shape}")

            fields = fields.permute(0, 1, 4, 2, 3)

            # Input -> Latent Representation Encoding
            preds = INN_E(fields)
            #print(preds.shape)
            # Latent Representation -> Point Cloud Decoding
            pc_recons = G(preds)


            loss_shape = chamfer_distance(pc.permute(0, 2, 1) + 0.5, pc_recons.permute(0, 2, 1) + 0.5, point_reduction='sum')[0]
            loss_shape = torch.mean(earth_mover_distance(pc.permute(0, 2, 1), pc_recons.permute(0, 2, 1), transpose=False))

            total_shape_loss += loss_shape.item() / args.n_points

            # Combined loss for the INN Encoder
            loss_inn_enc = loss_shape
            total_loss_inn_enc += loss_inn_enc.item()

            loss_inn_enc.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(INN_E.parameters(), args.clip)

            # Check grads

            E_opt.step()

            batch_count += 1

        E_scheduler.step()


        # Average Losses to report
        train_loss = total_loss_inn_enc / batch_count
        shape_recons_loss = total_shape_loss / batch_count

        outstr = 'Epoch %d Train Loss: %.7f' % (epoch, train_loss)
        outstr += '\nEpoch %d Shape Reconstruction Loss: %.7f' % (epoch, shape_recons_loss)
        print(outstr)

        ####################
        # Test
        ####################
        coef_loss = 0.0
        shape_loss = 0.0
        test_count = 0.0

        total_shape_loss = 0.0
        INN_E.eval()
        test_pred = []
        test_true = []
        batch_count = 0
        labels_recons = []
        for label, fields, pc in test_loader:

            pc, fields = pc.to(device, dtype=torch.float), fields.to(device, dtype=torch.float)


            pc = pc.permute(0, 2, 1)

            fields = fields.permute(0, 1, 4, 2, 3)
            batch_size = pc.size()[0]

            pc = pc.to(device)
            with torch.no_grad():
                preds = INN_E(fields)
                pc_recons = G(preds)
                labels_recons = label
                loss_shape = chamfer_distance(pc.permute(0, 2, 1) + 0.5, pc_recons.permute(0, 2, 1) + 0.5, point_reduction='sum')[0]
                total_shape_loss += loss_shape.item() / args.n_points

            test_count += batch_size
            batch_count += 1

        shape_loss = total_shape_loss / batch_count
        outstr = 'Epoch %d Val Shape Loss: %.6f' % (epoch, shape_loss)

        print(outstr)

        pc_recons = pc_recons.data.cpu().numpy()
        pc = pc.data.cpu().numpy()

        if shape_loss < best_loss:
          best_loss = shape_loss
          best_pc = pc
          best_pc_recons = pc_recons
          best_labels = labels_recons
          best_epoch = epoch
          best_model_dict = copy.deepcopy(INN_E.state_dict())
          best_opt_dict = copy.deepcopy(E_opt.state_dict())

        if epoch % args.save_freq == 0 and epoch != 0:
          #for k in range(min(20, batch_size)):
          #    fig = plot_PC_compare(best_pc[k][0], best_pc[k][1], best_pc[k][2], best_pc_recons[k][0], best_pc_recons[k][1], best_pc_recons[k][2], epoch=best_epoch)
          #    fig.savefig(
          #      join('/mnt/scratch/dikbayir/checkpoints',args.exp_name, 'samples', f'{epoch}_{k}.png'), dpi=300)
          #    plt.close(fig)

          # Save animations of test reconstructions
          for k in tqdm.tqdm(range(min(20, batch_size))):
              anim = animated_PC_compare(best_pc[k][0], best_pc[k][1], best_pc[k][2], best_pc_recons[k][0], best_pc_recons[k][1], best_pc_recons[k][2], title=best_labels[k])

              anim.save(
                  join(args.checkpoint_path, args.exp_name, 'samples', f'best_{k}.gif'), writer='imagemagick'
              )

          torch.save(best_model_dict, f'{args.checkpoint_path}%s/models/INN_E_best.t7' % args.exp_name)
          torch.save(best_opt_dict, f'{args.checkpoint_path}%s/models/E_opt_best.t7' % args.exp_name)
          time_best_epoch = time.time() - time_epoch0

        time_total = time.time() - time_epoch0

        # I use WANDB api to log the training stats each epoch. You can use the variable names down below to store training stats and dump them periodically or at the end of the training process
        # best_loss, best_epoch, time_best_epoch etc. are still being updated, you just need to save them.
        #
        wandb.log({'loss':train_loss, 'shape_loss_test': shape_loss, 'shape_loss_train': shape_recons_loss, 'epoch': epoch, 'lr': E_scheduler.get_last_lr()[0], 'best_loss': best_loss, 'best_epoch': best_epoch, 'time_best_epoch':time_best_epoch, 'time_total': time_total})



if __name__ == "__main__":
    # Training settings

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--save_freq', type=int, default=15)
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/scratch/dikbayir/checkpoints/')
    parser.add_argument('--ffd_norm', type=str, default=None)

    parser.add_argument('--G_path', type=str, default='./fixed_shapenet_airplanes_dataset/G_best.t7')

    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--data_root', type=str, default='./fixed_shapenet_airplanes_dataset/', help='Root path of the dataset.')
    parser.add_argument('--model', type=str, default='acsnet', metavar='N',
                        choices=['pointnet', 'dgcnn', 'pointneten', 'acsnet', 'acsnetsym'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=15000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
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
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--n_points', type=int, default=2048, help='Num points in the point-cloud.')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--grad_clip', action='store_true') # whether to use gradient clipping or not
    parser.add_argument('--clip', type=float, default=1.0) # if grad_clip is True, the value of the max gradient-norm to clip the gradient from. check out gradient clipping online for more details.

    parser.add_argument('--in_option', type=str, default='harm') # the input format, this is kind of a stupid parameter, it could be derived by the model directly. I will remove this soon
    parser.add_argument('--bn', action='store_true') # use batch-normalization or not, only has effect when the model is PointNet and DGCNN types

    parser.add_argument('--norm', type=str, default='none', choices=['none', 'minmax', 'std']) # input normalization type, for no normalization use 'none'
    parser.add_argument('--train_size', type=int, default=200) # the size of the training set
    parser.add_argument('--val_size', type=int, default=20) # the size of the validation set

    parser.add_argument('--in_size', type=int, default=2601) # number of complex harmonics to include in the input. in_size <= 2601, since we compute coefficients up to 50 harmonic degrees.
    parser.add_argument('--out_size', type=int, default=2601) # number of complex harmonics to include in the output. out_size <= 2601, since we compute coefficients up to 50 harmonic degrees.
                                                              # I really didn't change the number of harmonics in the output for the same frequency. You can change this depending on your case.

    parser.add_argument('--reduction', type=str, default='sum')
    parser.add_argument('--reconstruction_coef', type=float, default=1.0)
    parser.add_argument('--normal_mu', type=float, default=0.0)
    parser.add_argument('--normal_std', type=float, default=0.2)
    parser.add_argument('--use_bias', type=bool, default=True)
    parser.add_argument('--man_seed', type=int, default=42)
    parser.add_argument('--in_channels_inn', type=int, default=1)
    parser.add_argument('--n_theta', type=int, default=51)
    parser.add_argument('--n_phi', type=int, default=101)
    parser.add_argument('--class_obj', type=str, default='airplane')
    parser.add_argument('--fwd', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--inn_depth', type=int, default=3)
    parser.add_argument('--block_depth', type=int, default=3)
    parser.add_argument('--project_name', type=str, default="VINN_Pub")
    args = parser.parse_args()

    # Set up experiment name based on hyperparams
    exp_name_str = [args.exp_name, f'lr:{args.lr}', f'eps:{args.epochs}', f'n_sample:{args.n_points}', f'shape:{args.n_theta}x{args.n_phi}', f'inn_depth:{args.inn_depth}', f'block_depth:{args.block_depth}']
    args.exp_name = '_'.join(exp_name_str)

    _init_(args.checkpoint_path)

    #io = IOStream(f'{args.checkpoint_path}' + args.exp_name + '/run.log')
    #io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    #if args.cuda:
    #    io.cprint(
    #        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    #    torch.cuda.manual_seed(args.seed)
    #else:
    #    io.cprint('Using CPU')

    train(args)
    #debug_model(args)
