"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Class for rough motion estimation of the bot based on optical flow.
License: GNU General Public License v3.0
"""

import cv2
import numpy as np


class MotionEvaluator:
    def __init__(self, verbose=False):
        self.flow_estimator = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = None
        self.flow = None
        self.verbose = verbose

    def update(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
        else:
            self.flow = self.flow_estimator.calc(self.prev_gray, gray, None)
            self.prev_gray = gray

    def get_motion(self):
        if self.flow is not None:
            flow_x, flow_y = self.flow[..., 0], self.flow[..., 1]
            if np.sum(flow_y) > np.sum(flow_x) * 1.5:
                moving_forward = True
                if self.verbose:
                    print("MOVING FORWARD")
                # Note that flows are not normalized for comparability leading to saturation or black negative pixels:
                # cv2.imshow("flow x", flow_x)
                # cv2.imshow("flow y", flow_y)
            else:
                moving_forward = False
                if self.verbose:
                    print("NOT MOVING FORWARD")
            return moving_forward