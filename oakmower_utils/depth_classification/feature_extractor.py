"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Script for feature extraction (LBP histograms) for images in "clear_path" or "obstacle" directories
License: GNU General Public License v3.0
"""


import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import local_binary_pattern


depth_data_path = "../../../../oakmower_training_data/2020_10_26_flaeche_labeled/filtered"

visualize = True
apply_colormap = True
save_marked = False
stop_at_first = True

clear_path_imgs = []
obstacle_imgs = []
top_left = (137, 204)
bottom_right = (top_left[0] + 300, top_left[1] + 150)

# get and potentially visualize and/or save depth images including marks for relevant area and class (green/red)
for root, dirs, files in os.walk(depth_data_path):
    print(os.path.abspath(root))
    for file in sorted(files):
        if file.endswith(".png") and "marked" not in file:
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path, 0)
            if apply_colormap:
                img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            if visualize and not apply_colormap:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if "clear_path" in file_path:
                clear_path_imgs.append(np.copy(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]))
                marked_img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            elif "obstacle" in file_path:
                obstacle_imgs.append(np.copy(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]))
                marked_img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
            else:
                "depth image class not identfied"
                marked_img = cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)

            if visualize:
                cv2.imshow("marked depth image", marked_img)
                if stop_at_first:
                    cv2.waitKey(0)
                    stop_at_first = False
                else:
                    cv2.waitKey(50)
            if save_marked:
                if apply_colormap:
                    cv2.imwrite(file_path[:-4] + "_col_marked.png", marked_img)
                else:
                    cv2.imwrite(file_path[:-4] + "_marked.png", marked_img)

clear_path_lbps = []
obstacle_lbps = []
visualize_lbps = False

# get local binary patterns of relevant filtered depth image area:
for lbps, imgs, im_class in zip([clear_path_lbps, obstacle_lbps], [clear_path_imgs, obstacle_imgs], ["clear", "obstacle"]):
    for img in imgs:
        lbp = local_binary_pattern(img, 8, 1, "nri_uniform")
        lbps.append(np.copy(lbp))
        lbp = (lbp).astype(np.uint8)
        print(np.max(lbp))
        if visualize_lbps:
            cv2.imshow("lbp image" + im_class, cv2.applyColorMap(lbp, cv2.COLORMAP_TURBO))
            cv2.waitKey(50)
        lbps.append(lbp)

clear_features = []
obstacle_features = []

for lbps, features in zip([clear_path_lbps, obstacle_lbps], [clear_features, obstacle_features]):
    for lbp in lbps:
        hist, _ = np.histogram(lbp, density=True, bins=59, range=(0, 59))
        features.append(hist)

clear_features_df = pd.DataFrame(clear_features)
obstacle_features_df = pd.DataFrame(obstacle_features)

clear_features_df.to_csv("clear_features.csv")
obstacle_features_df.to_csv("obstacle_features.csv")