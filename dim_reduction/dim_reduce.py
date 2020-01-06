#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:51:32 2019

@author: abishekk
"""

#%% Setup - dimensionality reduction

import numpy as np
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#%% data
np.random.seed(42)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

#plotting
fig = plt.figure()
axes = fig.add_subplot(111,projection='3d')
axes.scatter(X[:,0],X[:,1],X[:,2],c='r',marker='o')
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')
plt.show()

#%% PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
print(pca.components_)
print(pca.explained_variance_ratio_)

plt.scatter(X2D[:,0],X2D[:,1],c='r',marker='x')
plt.show()

# PCA using explained variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(pca.components_)
print(pca.explained_variance_ratio_)
        