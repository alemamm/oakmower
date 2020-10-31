#!/usr/bin/env python3

"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Main file allowing for acquiring data for training as well as running and visualizing OAKMower, an application built to
show how the capabilities of the OpenCV Spatial AI Kit with depth (OAK-D) can be used to detect obstacle for
autonomous lawn mower navigation. Note that this does not include control of the bot itself.
Follow the example of the https://github.com/luxonis/depthai/depthai_demo.py (release 0.3.0) to build similar apps.
License: GNU General Public License v3.0
"""

import json
import os
from time import time, monotonic
import cv2
import numpy as np
import depthai
import tempfile


print('Using depthai module from: ', depthai.__file__)
print('Depthai version installed: ', depthai.__version__)

from depthai_helpers.version_check import check_depthai_version
check_depthai_version()

from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs

from oakmower_utils.pointcloud_projector import PointCloudProjector
from oakmower_utils.wls_filter import WLSFilter
from oakmower_utils.persister import Persister
from oakmower_utils.disparity_classifier import DisparityClassifier
from oakmower_utils.plane_classifier import PlaneClassifier
from oakmower_utils.motion_evaluator import MotionEvaluator


global args


class OAKMower:
    global is_rpi
    process_watchdog_timeout = 10 #seconds
    nnet_packets = None
    data_packets = None
    runThread = True
    logo = cv2.imread("oakmower_utils/OAKMower_logo_1280_left.png")

    show_single_windows = False
    show_overview = True

    status = {"plane": {"a": False, "b": False, "c": False, "d": False, },
              "disparity_clear": False,
              "objects_clear": False,
              "moving_forward": False}

    # overview frames:
    objects_frame = np.zeros([400, 640, 3], dtype=np.uint8)
    wls_frame = np.zeros([400, 640, 3], dtype=np.uint8)
    depth_frame = np.zeros([400, 640, 3], dtype=np.uint8)
    rectified_right_frame = np.zeros([400, 640, 3], dtype=np.uint8)

    font = cv2.FONT_HERSHEY_DUPLEX

    def label_imgs(self, frame, label):
        frame = cv2.putText(frame, label, (10, 385), self.font, 1, (255, 255, 255), 2)
        return frame

    def visualize_classifications(self):
        augmented_header = np.copy(self.logo)

        # movement:
        if self.status["moving_forward"]:
            augmented_header = cv2.putText(augmented_header, "IS", (530, 175), self.font, 1, (255, 255, 255), 2)
        else:
            augmented_header = cv2.putText(augmented_header, "IS NOT", (530, 175), self.font, 1, (255, 255, 255), 2)
        augmented_header = cv2.putText(augmented_header, "MOVING FORWARD", (650, 175), self.font, 1, (255, 255, 255), 2)

        # clear path?
        if self.status["objects_clear"] is True and self.status["disparity_clear"] is True and True in self.status["plane"].values():
            path_color = (0, 255, 0)
        else:
            path_color = (0, 0, 255)
        augmented_header = cv2.rectangle(augmented_header, (570, 50), (900, 95), path_color, -1)
        augmented_header = cv2.putText(augmented_header, "CLEAR PATH AHEAD", (580, 85), self.font, 1, (0, 0, 0), 2)

        # object detection classifier:
        if self.status["objects_clear"] is True:
            objects_color = (0, 255, 0)
        else:
            objects_color = (0, 0, 255)
        augmented_header = cv2.rectangle(augmented_header, (960, 20), (1260, 65), objects_color, -1)
        augmented_header = cv2.putText(augmented_header, "OBJECTS", (1040, 55), self.font, 1, (0, 0, 0), 2)

        # disparity classifier:
        if self.status["disparity_clear"] is True:
            disparity_color = (0, 255, 0)
        else:
            disparity_color = (0, 0, 255)
        augmented_header = cv2.rectangle(augmented_header, (960, 80), (1260, 125), disparity_color, -1)
        augmented_header = cv2.putText(augmented_header, "DISPARITY", (1030, 115), self.font, 1, (0, 0, 0), 2)

        # segmented plane (point cloud) classifier:
        if self.status["plane"]["a"] or self.status["plane"]["b"] or self.status["plane"]["c"] or self.status["plane"]["d"] is True:
            disparity_color = (0, 255, 0)
        else:
            disparity_color = (0, 0, 255)
        augmented_header = cv2.rectangle(augmented_header, (960, 140), (1260, 185), disparity_color, -1)
        augmented_header = cv2.putText(augmented_header, "POINT CLOUD", (1010, 175), self.font, 1, (0, 0, 0), 2)

        return augmented_header

    def reset_process_wd(self):
        global wd_cutoff
        wd_cutoff = monotonic()+self.process_watchdog_timeout
        return

    def stop_loop(self):
        self.runThread = False

    def start_loop(self):
        top_left = (137, 204)
        width = 300
        height = 150
        bottom_right = (top_left[0] + width, top_left[1] + height)

        mode = "both"  # "3dvis", "wls" or "both"
        persist = False
        output_path = "../../oakmower_training_data"
        persister = Persister(output_path)
        depth_classifier = DisparityClassifier(top_left, width, height)
        plane_classifier = PlaneClassifier()
        motion_evaluator = MotionEvaluator()

        if mode == "3dvis":
            streams = ["disparity", 'rectified_right']
            fit_plane = True
            filter_wls = False
        elif mode == "wls":
            streams = ["disparity", 'rectified_right']
            fit_plane = False
            filter_wls = True
        elif mode == "both":
            streams = ['previewout', 'metaout', "disparity", 'rectified_right', ]
            fit_plane = True
            filter_wls = True
        else:
            streams = ["rectified_right"]
            fit_plane = False
            filter_wls = False
            print("no mode selected")

        wls_filter = WLSFilter(persister=persister, show_single_windows=self.show_single_windows)

        rectified_right = None
        pcl_projector = None
        detections = []

        cliArgs = CliArgs()
        args = vars(cliArgs.parse_args())

        configMan = DepthConfigManager(args)

        config = configMan.jsonConfig
        config["streams"] = streams
        config["camera"]["mono"]["resolution_h"] = 400
        config["camera"]["mono"]["fps"] = 10
        config["camera"]["rgb"]["fps"] = 10

        self.device = depthai.Device(args['device_id'], False)
        p = self.device.create_pipeline(config=config)

        if p is None:
            print('Pipeline is not created.')
            exit(3)

        self.reset_process_wd()

        while self.runThread:
            nnet_packets, data_packets = p.get_available_nnet_and_data_packets(True)

            packets_len = len(nnet_packets) + len(data_packets)
            if packets_len != 0:
                self.reset_process_wd()
            else:
                cur_time = time.monotonic()
                if cur_time > wd_cutoff:
                    print("process watchdog timeout")
                    os._exit(10)

            for nnet_packet in nnet_packets:
                detections = list(nnet_packet.getDetectedObjects())

            for packet in data_packets:

                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    data0 = data[0, :, :]
                    img_h = data0.shape[0]
                    img_w = data0.shape[1]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    alpha = 0.3
                    beta = (1.0 - alpha)
                    red_im = np.ones_like(data2) * 255
                    data2[int(img_h/2):,:] = cv2.addWeighted(data2[int(img_h/2):,:], alpha, red_im[int(img_h/2):,:], beta, 0.0)
                    frame = cv2.merge([data0, data1, data2])
                    self.status["objects_clear"] = True
                    for detection in detections:
                        pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                        pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)
                        if pt2[1] < int(img_h/2) or ((pt2[0] - pt1[0]) > 0.95*img_w) or ((pt2[1] - pt1[1]) > 0.95*img_h):
                            cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                            self.status["objects_clear"] = False
                    objects_frame = np.zeros([400, 640, 3], dtype=np.uint8)
                    objects_frame[54:354, 137:437] = frame
                    if self.show_single_windows:
                        cv2.imshow('cockpit', objects_frame)
                    self.objects_frame = self.label_imgs(objects_frame, "Object detection")

                    if persister is not None:
                        persister.add_preview_bb_free(frame)

                if packet.stream_name == "rectified_right":
                    rectified_right = packet.getData()
                    rectified_right = cv2.flip(rectified_right, flipCode=1)
                    motion_evaluator.update(np.copy(rectified_right[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]))
                    self.status["moving_forward"] = motion_evaluator.get_motion()
                    rect_right_bb = wls_filter.update(rectified_right, packet.stream_name)
                    if rect_right_bb is not None:
                        self.rectified_right_frame = self.label_imgs(cv2.cvtColor(rect_right_bb, cv2.COLOR_GRAY2BGR), "Right camera")

                elif packet.stream_name == "depth" or packet.stream_name == "disparity":

                    if packet.stream_name == "disparity":
                        disparity_frame = packet.getData()
                        if fit_plane:
                            frame_f64 = disparity_frame.astype(np.float64)
                            frame_f64[frame_f64 < 1] = 0.000001
                            depth = 75 * 883.14 / frame_f64
                            frame16 = depth.astype(np.uint16)
                            frame16[frame16 < 1] = 65535
                            depth_frame = frame16
                        else:
                            depth_frame = np.zeros_like(disparity_frame)
                    else:
                        depth_frame = packet.getData()
                        disparity_frame = None

                    if filter_wls:
                        wls_filter.update(frame=disparity_frame, stream_name=packet.stream_name)

                    if fit_plane:
                        if rectified_right is not None:
                            if pcl_projector is None:
                                fd, path = tempfile.mkstemp(suffix='.json')
                                with os.fdopen(fd, 'w') as tmp:
                                    json.dump({
                                        "width": 600,
                                        "height": 400,
                                        "intrinsic_matrix": [item for row in self.device.get_right_intrinsic() for item in
                                                             row]
                                    }, tmp)
                                pcl_projector = PointCloudProjector(path, persister=persister)
                            plane_parameters = pcl_projector.rgbd_to_projection(depth_frame,
                                                                                np.ones_like(rectified_right))
                            self.status["plane"] = plane_classifier.classify(plane_parameters)
                            pcl_projector.visualize_pcd()

                    depth_frame8 = (65535 // depth_frame).astype(np.uint8)
                    # colorize depth map, comment out code below to obtain grayscale
                    depth_frame8 = cv2.applyColorMap(depth_frame8, cv2.COLORMAP_HOT)
                    depth_frame8 = cv2.rectangle(depth_frame8, top_left, bottom_right, (255, 255, 255), 2)
                    if self.show_single_windows:
                        cv2.imshow("depth", depth_frame8)
                    self.depth_frame = self.label_imgs(depth_frame8, "Depth")

                if filter_wls:
                    colored_wls = wls_filter.filter()
                    if colored_wls is not None:
                        self.wls_frame = self.label_imgs(colored_wls, "Disparity (filtered using right camera)")
                    if wls_filter.filtered_disp is not None:
                        self.status["disparity_clear"] = depth_classifier.classify(wls_filter.filtered_disp)

                if self.show_overview:
                    comb_frame = np.vstack([self.visualize_classifications(),
                                            np.hstack([self.objects_frame, self.wls_frame]),
                                           np.hstack([self.depth_frame, self.rectified_right_frame])])
                    cv2.imshow("overview", comb_frame)

            print(self.status)

            if cv2.waitKey(1) == ord("s"):
                print("Writing to disk")
                persister.write_to_disk()
                persister.entries.clear()
            if cv2.waitKey(1) == ord("c"):
                persister.entries.clear()
            if cv2.waitKey(1) == ord("p"):
                if persist:
                    print("Accumulating data stopped")
                    persist = False
                else:
                    print("Accumulating data")
                    persist = True
            if cv2.waitKey(1) == ord("q"):
                break

        del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
        del self.device


if __name__ == "__main__":
    dai = OAKMower()
    dai.start_loop()
