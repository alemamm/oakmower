"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Weighted least squares filtering class combining disparity and gray scale stream data for robust, edge-preserving data.
Check: https://github.com/luxonis/depthai-experiments/tree/master/wls-filter for an adjustable stand-alone demo.
License: GNU General Public License v3.0
"""


import cv2
import numpy as np


class WLSFilter():
    def __init__(self, top_left=(137, 204), width=300, height=150, persister=None, show_single_windows=False):
        self.persister = persister
        self.top_left = top_left
        self.bottom_right = (top_left[0] + width, top_left[1] + height)
        self.prev_right = None
        self.prev_left = None
        self.prev_disp = None
        self.filtered_disp = None
        self.colored_wls = None
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        self.filter_lamda = 8000
        self.filter_sigma = 1.5
        self.wls_stream = "wls_filter"
        self.show_single_windows = show_single_windows
        #self.window = cv2.namedWindow(self.wls_stream)

    def update(self, frame, stream_name):
        if stream_name == 'rectified_right':
            self.prev_right = frame
            rect_right_frame = cv2.rectangle(np.copy(self.prev_right), self.top_left, self.bottom_right, (255, 255, 255), 2)
            if self.show_single_windows:
                cv2.imshow(stream_name, rect_right_frame)
            return rect_right_frame
        elif stream_name == 'disparity':
            self.prev_disp = frame
            #cv2.imshow(stream_name, frame)
            return None

    def filter(self):
        self.filtered_disp = None
        if self.prev_right is not None:
            if self.prev_disp is not None:
                # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
                self.wls_filter.setLambda(self.filter_lamda)
                # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
                self.wls_filter.setSigmaColor(self.filter_sigma)
                self.filtered_disp = self.wls_filter.filter(self.prev_disp, self.prev_right)
                #cv2.imshow(self.wls_stream, self.filtered_disp)
                cv2.normalize(self.filtered_disp, self.filtered_disp, 0, 255, cv2.NORM_MINMAX)
                self.colored_wls = cv2.applyColorMap(self.filtered_disp, cv2.COLORMAP_TURBO)
                self.colored_wls = cv2.rectangle(self.colored_wls, self.top_left, self.bottom_right, (255, 255, 255), 2)
                if self.show_single_windows:
                    cv2.imshow(self.wls_stream + "_color", self.colored_wls)
                if self.persister is not None:
                    self.persister.add_filtered(np.copy(self.filtered_disp))
                    self.persister.add_prev_rect_right(np.copy(self.prev_right))
                self.prev_right = None
                self.prev_left = None
                self.prev_disp = None
        return self.colored_wls
