"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Script fit, evaluate and save anomaly detection models to be applied for plane/point cloud classification
Also check: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py
License: GNU General Public License v3.0
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib


predict_only = False

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

outliers_fraction = 0.15

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction)),
#    ("Local Outlier Factor", LocalOutlierFactor(contamination=outliers_fraction))
    ]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-2, 2, 150),
                     np.linspace(-2, 2, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 5, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

plot_num = 1

a, b, c, d, pcl_sizes = [], [], [], [], []

# replace by folder containing persisted .json plane files
plane_data_path = "../../../../oakmower_training_data/2020_10_26_flaeche"
print(os.path.abspath(plane_data_path))

for root, dirs, files in os.walk(plane_data_path):
    for file in files:
        if file.endswith(".json"):
            f = open(os.path.join(root, file))
            data = json.load(f)
            pcl_sizes.append(data["pcl_size"])
            a.append(data["a"])
            b.append(data["b"])
            c.append(data["c"])
            d.append(data["d"])

datasets = [
    np.array([pcl_sizes, a]).transpose(),
    np.array([pcl_sizes, b]).transpose(),
    np.array([pcl_sizes, c]).transpose(),
    np.array([pcl_sizes, d]).transpose()]

dataset_names = ["a", "b", "c", "d"]

for i_dataset, X in enumerate(datasets):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaler_filename = "models/" + dataset_names[i_dataset] + "_scaler.save"
    joblib.dump(scaler, scaler_filename)

    for name, algorithm in anomaly_algorithms:
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        model_filename = dataset_names[i_dataset] + "_" + name + "_" + str(outliers_fraction) + ".save"
        if i_dataset == 0:
            plt.title(name, size=18)

        if predict_only:
            # load fitted model for prediction - not used for Local Outlier Factor, since no novelties are expected
            algorithm = joblib.load(model_filename)
            y_pred = algorithm.fit(X).predict(X)
        else:
            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)

        joblib.dump(algorithm, model_filename)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=1, colors='black')

        colors = np.array(['#eb4634', '#29cf50'])
        plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[(y_pred + 1) // 2])

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        #plt.xlabel("Point cloud size / 10^5")
        #plt.ylabel(dataset_names[i_dataset])
        plot_num += 1

plt.savefig("anomaly_detection.png")
plt.show()