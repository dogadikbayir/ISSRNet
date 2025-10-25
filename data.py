#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

def add_complex_awgn(signal, snr_db):
    """Adds complex additive white Gaussian noise to a complex signal.

    Args:
        signal (numpy.ndarray): The input complex signal.
        snr_db (float): The desired signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: The noisy complex signal.
    """
    sig_avg_power = np.mean(np.abs(signal)**2)
    sig_avg_db = 10 * np.log10(sig_avg_power)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_power = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    std_noise = np.sqrt(noise_avg_power / 2)

    np.random.seed(42)

    complex_noise = np.random.normal(mean_noise, std_noise, size=signal.shape)

    np.random.seed(42)
    complex_noise += 1j * np.random.normal(mean_noise, std_noise, size=signal.shape)

    noisy_signal = signal + complex_noise
    return noisy_signal

def add_noise(signal, snr=10, seed=42):
    '''Adds random Gaussian noise with SNR to complex sinal'''


    assert len(signal.shape) == 3  # needs to be (N, 2)
    n_theta, n_phi = signal.shape[0], signal.shape[1]
    signal = signal.reshape((signal.shape[0]*signal.shape[1], signal.shape[2]))
    signal = signal[..., 0] + 1j*signal[..., 1]  # first, convert 2d vector to complex signal

    sig_avg_power = np.mean(np.abs(signal)**2)
    sig_avg_db = 10 * np.log10(sig_avg_power)
    noise_avg_db = sig_avg_db - snr
    noise_avg_power = 10 ** (noise_avg_db / 20)
    mean_noise = 0
    #std_noise = np.sqrt((10**(-snr/20)/(np.sqrt(n_theta*n_phi))))
    std_noise = np.sqrt(noise_avg_power / 2)
    np.random.seed(seed)
    complex_noise = np.random.normal(mean_noise, std_noise, size=signal.shape)
    np.random.seed(seed)
    complex_noise = complex_noise + 1j * np.random.normal(mean_noise, std_noise, size=signal.shape)
    noisy_signal = signal + signal*complex_noise

    return noisy_signal.view(float).reshape(n_theta, n_phi, 2)


def convert2db(field):

    field = field[..., 0] + 1j*field[..., 1]
    # We convert the complex pressure field into dB real magnitude
    dB_field = 10*np.log10(4*np.pi * np.abs(field)**2)

    return dB_field


def load_airplane_multidir_ordered(data_root, args, isMC=False, n_theta=51, n_phi=101, pc_sample=2048, pc_sample_out=2048, snr=-1):

    if pc_sample_out != pc_sample:
        ordered_pcs_out = np.load(f"{data_root}/pcs_{pc_sample_out}.npy")
    else:
        pc_sample_out = -1

    ordered_pcs = np.load(f"{data_root}/pcs_{pc_sample}.npy")
    field_dict = np.load(f"{data_root}/fields_dict_{n_theta}x{n_phi}.npy", allow_pickle=True)

    sorted_dict = sorted(field_dict.item().items())

    ordered_labels = [t[0] for t in sorted_dict]

    # If multidir, 1 channel, shape =>

    #Convert to dB
    print("Creating Dataset")
    if not isMC:
        ordered_fields = [np.array(t[1]).reshape((np.array(t[1]).shape[0], n_theta, n_phi, 2)) for t in sorted_dict]
        # Convert to dB
        ordered_fields_db = []
        for i in range(len(ordered_fields)):
            ordered_fields_db.append([])
            for j in range(len(ordered_fields[i])):

                field_db = ordered_fields[i][j]
                field_db_no_noise = convert2db(field_db)
                #plt.imshow(field_db_no_noise)
                #plt.savefig(f"field_no_noise.png", dpi=600)
                #plt.close('all')
                #add noise
                if snr > -10:
                    field_db = add_noise(ordered_fields[i][j], snr=snr, seed=i*j)

                field_db = convert2db(field_db)

                #plt.imshow(field_db)
                #plt.savefig(f"field_noisy_{snr}.png", dpi=600)
                #plt.close('all')

                field_db = field_db.reshape((field_db.shape[0], field_db.shape[1], 1))
                ordered_fields_db[i].append(field_db)
            #exit(0)
            ordered_fields_db[i] = np.array(ordered_fields_db[i])

        ordered_fields = ordered_fields_db
    else:
        ordered_fields = [np.array(t[1]).reshape((n_theta, n_phi, np.array(t[1]).shape[0])) for t in sorted_dict]

    if pc_sample_out != -1:
        return ordered_labels, ordered_pcs, ordered_fields, ordered_pcs_out

    ordered_fieldss = []
    for field in ordered_fields:
        if field.shape == (6, n_theta, n_phi, 1):
            ordered_fieldss.append(field)
        else:
            field.shape

    ordered_fields = np.array(ordered_fieldss)

    print(ordered_fields.shape)
    return ordered_labels, ordered_pcs, ordered_fields

class ACS_Dataset_ordered(Dataset):

    def __init__(self, ordered_labels, ordered_X, ordered_y):

        self.X = ordered_X  # X are the inputs

        self.X = np.array(self.X)  # convert to ndarray
        print(self.X.shape)
        if not torch.is_tensor(self.X):
            self.X = torch.from_numpy(self.X)
        self.y = ordered_y  # y are the point-clouds
        if not torch.is_tensor(self.y):
            self.y = torch.from_numpy(self.y)

        self.labels = ordered_labels
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx, add_noise=False):

        return self.labels[idx], self.X[idx], self.y[idx]  # here, idx is common for all three containers, since we order everything up previously
