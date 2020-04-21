#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

gradient = 'summer'

plt.title("Mountain Elevation")

cmap = plt.get_cmap(gradient)

sca = plt.scatter(x, y, c=z, vmin=0, vmax=z.max(), cmap=cmap)
plt.colorbar(sca, label="elevation (m)")
plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")

plt.show()
