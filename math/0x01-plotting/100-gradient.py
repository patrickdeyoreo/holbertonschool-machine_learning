#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

gradient = 'summer'

scm = plt.cm.ScalarMappable(cmap=plt.get_cmap('summer'))
scm.set_clim([0, z.max()])

fig, axs = plt.subplots()
fig.suptitle("Mountain Elevation")
fig.colorbar(scm, label="elevation (m)")

axs.scatter(x, y, c=z, cmap=scm.get_cmap())
axs.set_xlabel("x coordinate (m)")
axs.set_ylabel("y coordinate (m)")

plt.show()
