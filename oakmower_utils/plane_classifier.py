"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Class for outlier estimation of parameters of segmented plane
License: GNU General Public License v3.0
"""

import numpy as np
import joblib
import time
from termcolor import colored

model_paths = ["oakmower_utils/anomaly_detection/models/a_Robust covariance_0.25.save",
               "oakmower_utils/anomaly_detection/models/b_Robust covariance_0.25.save",
               "oakmower_utils/anomaly_detection/models/c_Robust covariance_0.25.save",
               "oakmower_utils/anomaly_detection/models/d_Robust covariance_0.25.save",
               ]

scaler_paths = ["oakmower_utils/anomaly_detection/models/a_scaler.save",
               "oakmower_utils/anomaly_detection/models/b_scaler.save",
               "oakmower_utils/anomaly_detection/models/c_scaler.save",
               "oakmower_utils/anomaly_detection/models/d_scaler.save",
               ]

class PlaneClassifier():
    def __init__(self, scaler_path=scaler_paths, model_paths=model_paths, verbose=False):
        self.scalers = self.load_sk_objects(scaler_paths)
        self.models = self.load_sk_objects(model_paths)
        self.verbose = verbose

    def load_sk_objects(self, sk_object_paths):
        sk_objects = {}
        for param, sk_object_path in zip(["a", "b", "c", "d"], sk_object_paths):
            sk_objects[param] = joblib.load(sk_object_path)
        return sk_objects

    def classify(self, plane_parameters):
        #results = {"a": None, "b": None, "c": None, "d": None}
        results = []
        for param in ["a", "b", "c", "d"]:
            features = np.array([plane_parameters["pcl_size"], plane_parameters[param]]).reshape(1, -1)
            features = self.scalers[param].transform(features)
            #print(features)
            result = self.models[param].predict(features)[0]
            if result < 0:
                results.append(False)
            else:
                results.append(True)
        if self.verbose:
            if max(results) > 0:
                print(colored('PLANE', 'green'))
            else:
                print(colored('PLANE', 'red'))
        return {"a": results[0], "b": results[1], "c": results[2], "d": results[3]}