"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
t-SNE clustering of features to check suitability of features
License: GNU General Public License v3.0
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.preprocessing import StandardScaler

clear_features_df = pd.read_csv("clear_features.csv")
obstacle_features_df = pd.read_csv("obstacle_features.csv")

clear_features = clear_features_df.values[:, 1:]
obstacle_features = obstacle_features_df.values[:, 1:]

clear_color = np.zeros_like(clear_features[:, 1])
obstacle_color = np.ones_like(obstacle_features[:, 1])

X = np.vstack([clear_features, obstacle_features])
color = np.hstack([clear_color, obstacle_color])
scaler = StandardScaler()
X = scaler.fit_transform(X)

fig = plt.figure()
fig.suptitle("t-SNE", fontsize=10)

t_SNE = manifold.TSNE(n_components=2, init='random', random_state=42)

green_red_cmap = colors.ListedColormap(['g', 'r'])

Y = t_SNE.fit_transform(X)
ax = fig.add_subplot()
ax.scatter(Y[:, 0], Y[:, 1], c=color, s=9, cmap=green_red_cmap)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')

plt.savefig("t_sne.png")
plt.show()