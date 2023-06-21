# coding: utf-8
# Script for performing text language detection
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
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot
from utils.functions import CPD_NDE, makedir

# parameter settings
swl = 64 # sliding window length 
hidden_size = [32, 32, 32]
ML_epochs = 20
VCL_epochs = 1
p_lambda = 5
single_head = True
single_test = False
batch_size = None # batch size (should be smaller than 2*swl)

# Load data
Y = np.load("./baselines/data/Language Detection.npy").transpose()
nougat = np.load("./baselines/data/nougat_language.npy")
newma = np.load("./baselines/data/newma_language.npy")
knn = np.load("./baselines/data/knn_language.npy")

# Change point detection
start = time.time()
stat = CPD_NDE(Y, swl, hidden_size, ML_epochs, VCL_epochs, p_lambda, single_head, single_test, batch_size)
end = time.time()
print('Running time: ', end - start)

# alignment for baselines
T = np.shape(Y)[0]
stat = stat[2*swl:]
knn = knn[1:]
newma = newma[1:]
# draw figures
color = "#2F7FC1"
fig = plt.figure(figsize = (6, 5.5), dpi = 120)
plt.subplot(4, 1, 1)
plt.plot(range(2*swl, T), knn, color = color)
plt.axvline(1014, color = '#FFBE7A')
plt.axvline(1014 + swl, color = "#FA7F6F")
plt.axvline(1608, color = '#FFBE7A')
plt.axvline(1608 + swl, color = "#FA7F6F")
plt.legend(['kNN'], loc = 2)
plt.subplot(4, 1, 2)
plt.plot(range(2*swl, T), newma, color = color)
plt.axvline(1014, color = '#FFBE7A')
plt.axvline(1014 + swl, color = "#FA7F6F")
plt.axvline(1608, color = '#FFBE7A')
plt.axvline(1608 + swl, color = "#FA7F6F")
plt.legend(['MA'], loc = 2)
plt.subplot(4, 1, 3)
plt.plot(range(2*swl, T), np.abs(nougat), color = color)
plt.axvline(1014, color = '#FFBE7A')
plt.axvline(1014 + swl, color = "#FA7F6F")
plt.axvline(1608, color = '#FFBE7A')
plt.axvline(1608 + swl, color = "#FA7F6F")
plt.legend(['NOUGAT'], loc = 2)
plt.subplot(4, 1, 4)
plt.plot(range(2*swl, T), stat, color = color)
plt.axvline(1014, color = '#FFBE7A')
plt.axvline(1014 + swl, color = "#FA7F6F")
plt.axvline(1608, color = '#FFBE7A')
plt.axvline(1608 + swl, color = "#FA7F6F")
plt.legend(['NODE'], loc = 2)
plt.tight_layout()
plt.subplots_adjust(hspace = 0.28)
plt.savefig("./figures/language.pdf", bbox_inches='tight')
plt.show()