# coding: utf-8
# Script for performing change point detection on simulated data
#
# Reference: 
# xxx
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard
#
# 2022/09
# Implemented by
# Xiuheng Wang, Ricardo Borsoi
# xiuheng.wang@oca.eu, raborsoi@gmail.com

from __future__ import division
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.functions import CPD_NDE, makedir
import multiprocessing
print('the number of CPU cores: ', multiprocessing.cpu_count())
from multiprocessing import Pool
from functools import partial
import h5py

# parameter settings
swl = 64 # sliding window length 
hidden_size = [16, 16, 16]
ML_epochs = 20
VCL_epochs = 1
p_lambda = 25
single_head = True
single_test = False
batch_size = None # batch size (should be smaller than 2*swl)

# Load data
with h5py.File("./baselines/data/data.h5", "r") as f:
    data = f.get('data')
    Y_all  = np.array(data)

# define partial function
pfunc = partial(CPD_NDE, swl = swl, hidden_size = hidden_size, ML_epochs = ML_epochs, VCL_epochs = VCL_epochs, \
	p_lambda = p_lambda, single_head = single_head, single_test = single_test, batch_size = batch_size)

# change point detection
if __name__ == '__main__':
	pool = Pool(multiprocessing.cpu_count()) # Create a multiprocessing Pool
	STAT = pool.map(pfunc, Y_all)
	pool.close()
	pool.join()
	# save data
	np.save("./data/nde.npy", STAT)
