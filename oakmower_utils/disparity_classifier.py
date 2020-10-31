"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Class for prediction of WLS-filtered disparity and rectified_right stream data using a RBF kernel
Support Vector Machine given histograms of non-rotation-invariant uniform local binary patterns as inputs.
License: GNU General Public License v3.0
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
import time
from termcolor import colored

clf_path = "oakmower_utils/depth_classification/svc_rbf_10.save"
scaler_path = "oakmower_utils/depth_classification/scaler.save"


class DisparityClassifier():
    def __init__(self, top_left=(137, 204), width=300, height=150,
                 scaler_path=scaler_path, clf_path=clf_path, verbose=False):
        self.top_left = top_left
        self.bottom_right = (top_left[0] + width, top_left[1] + height)
        self.scaler = joblib.load(scaler_path)
        self.classifier = joblib.load(clf_path)
        self.verbose = verbose

    def get_features(self, wls_filtered):
        cropped = np.copy(wls_filtered[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]])
        lbp = local_binary_pattern(cropped, 8, 1, "nri_uniform")
        hist, _ = np.histogram(lbp, density=True, bins=59, range=(0, 59))
        features = hist
        return features

    def classify(self, wls_filtered):
        features = self.get_features(wls_filtered)
        features = self.scaler.transform(features.reshape(1, -1))
        result = self.classifier.predict(features)
        if result[0] == 0.0:
            disparity_clear = True
            if self.verbose:
                print(colored('WLS', 'green'))
        else:
            disparity_clear = False
            if self.verbose:
                print(colored('WLS', 'red'))
        return disparity_clear