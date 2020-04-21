#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
fig.suptitle("PCA of Iris Dataset")

axs = fig.add_subplot(111, projection='3d')
axs.set_xlabel('U1')
axs.set_ylabel('U2')
axs.set_zlabel('U3')
axs.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels,
            cmap=plt.cm.get_cmap('plasma'))

plt.show()
