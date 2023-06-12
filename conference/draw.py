import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils.functions import comp_roc

swl = 64
tr = 400 - 2 * swl

# load data
nde = np.load("./data/node.npy").transpose()
with h5py.File("./baselines/data/nougat.h5", "r") as f:
    data = f.get('nougat')
    nougat  = np.array(data).transpose()
with h5py.File("./baselines/data/newma.h5", "r") as f:
    data = f.get('newma')
    newma  = np.array(data).transpose()
with h5py.File("./baselines/data/knn.h5", "r") as f:
    data = f.get('knn')
    knn  = np.array(data).transpose()

# alignment for baselines
nde = nde[2*swl:, :]
knn = knn[1:, :]
newma = newma[1:, :]

# draw figures
color = "#2F7FC1"
fig = plt.figure(figsize = (6, 5.5), dpi = 120)
ax = fig.add_subplot(4, 1, 1)
avg = np.mean(knn, axis = 1)
std = np.std(knn, axis = 1)
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(range(2*swl, 700), avg, color = color)
ax.fill_between(range(2*swl, 700), r1, r2, alpha=0.2)
plt.axvline(400, color = '#FFBE7A')
plt.axvline(400 + swl, color = "#FA7F6F")
plt.legend(['kNN'], loc = 2)
ax = fig.add_subplot(4, 1, 2)
avg = np.mean(newma, axis = 1)
std = np.std(newma, axis = 1)
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(range(2*swl, 700), avg, color = color)
ax.fill_between(range(2*swl, 700), r1, r2, alpha=0.2)
plt.axvline(400, color = '#FFBE7A')
plt.axvline(400 + swl, color = "#FA7F6F")
plt.legend(['MA'], loc = 2)
ax = fig.add_subplot(4, 1, 3)
avg = np.mean(nougat, axis = 1)
std = np.std(nougat, axis = 1)
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(range(2*swl, 700), avg, color = color)
ax.fill_between(range(2*swl, 700), r1, r2, alpha=0.2)
plt.axvline(400, color = '#FFBE7A')
plt.axvline(400 + swl, color = "#FA7F6F")
plt.legend(['NOUGAT'], loc = 2)
ax = fig.add_subplot(4, 1, 4)
avg = np.mean(nde, axis = 1)
std = np.std(nde, axis = 1)
r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
ax.plot(range(2*swl, 700), avg, color = color)
ax.fill_between(range(2*swl, 700), r1, r2, alpha=0.2)
plt.axvline(400, color = '#FFBE7A')
plt.axvline(400 + swl, color = "#FA7F6F")
plt.legend(['NODE'], loc = 2)
plt.tight_layout()
plt.subplots_adjust(hspace = 0.28)
plt.savefig("./figures/simulation.pdf", bbox_inches='tight')
# plt.show()

# draw ROC curves
fig = plt.figure(figsize = (3.8, 3.6), dpi = 120)
pfa, pd = comp_roc(knn, tr, N = 100)
plt.plot(pfa, pd, color="#8ECFC9", label='kNN')
pfa, pd = comp_roc(newma, tr, N = 100)
plt.plot(pfa, pd, color="#82B0D2", label='MA')
pfa, pd = comp_roc(nougat, tr, N = 100)
plt.plot(pfa, pd, color="#FFBE7A", label='NOUGAT')
pfa, pd = comp_roc(nde, tr, N = 100)
plt.plot(pfa, pd, color="#FA7F6F", label='NODE')
plt.xlabel("False alarm rate")
plt.ylabel("Detection rate")
plt.legend() 
plt.tight_layout()
plt.savefig("./figures/roc.pdf", bbox_inches='tight')
plt.show()