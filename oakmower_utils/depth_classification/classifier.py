"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Script for train, evaluate and save SVM.
License: GNU General Public License v3.0
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

mode = "fit_persist" # fit_persist or compare

clear_features_df = pd.read_csv("clear_features.csv")
obstacle_features_df = pd.read_csv("obstacle_features.csv")

clear_features = clear_features_df.values[:, 1:]
obstacle_features = obstacle_features_df.values[:, 1:]

clear_y = np.zeros_like(clear_features[:, 1])
obstacle_y = np.ones_like(obstacle_features[:, 1])

X = np.vstack([clear_features, obstacle_features])
y = np.hstack([clear_y, obstacle_y])

scaler = StandardScaler()
X = scaler.fit_transform(X)

cross_val_scores = []
kernel = "rbf"
if mode == "compare":
    Cs = [0.01, 0.1, 1, 10, 100]
elif mode == "fit_persist":
    Cs = [10]
else:
    Cs = None

for C in Cs:
    clf = SVC(kernel=kernel, C=C).fit(X, y)
    scores = cross_val_score(clf, X, y, cv=20, scoring='f1')
    cross_val_scores.append(scores)
    print(np.median(scores), scores)
    if mode == "fit_persist":
        scaler_filename = "scaler.save"
        joblib.dump(scaler, scaler_filename)
        model_filename = "svc_" + kernel + "_" + str(Cs[0]) + ".save"
        joblib.dump(clf, model_filename)

plt.boxplot(cross_val_scores)
plt.xticks([1, 2, 3, 4, 5], ["0.01", "0.1", "1", "10", "100"])
plt.title("SVC - " + kernel + " kernel (20 cross-validations)")
plt.ylabel("F1 score")
plt.xlabel("C")
plt.show()