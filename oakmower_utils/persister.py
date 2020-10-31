"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Class for persisting data for labeling and/or training.
License: GNU General Public License v3.0
"""

import time
import json
import cv2
import numpy as np
import open3d as o3d
import os


class Persister():
    def __init__(self, output_path):
        self.entries = []
        self.output_path = output_path

    def write_to_disk(self):
        print(len(self.entries))
        for entry in self.entries:
            print(entry)
            entry_type = entry["entry_type"]
            time = str(entry["time"])
            file_path = os.path.join(self.output_path, time + "_" + entry_type)
            file_path.replace(".","-")
            if entry_type == "plane":
                with open(file_path + ".json", 'w') as outfile:
                    json.dump(entry, outfile)
            elif entry_type == "filtered" or "prev_rect_right" or "preview_bb_free":
                image = entry["image"]
                cv2.imwrite(file_path + ".png", image)
            elif entry_type == "screen_buffer":
                screen_buffer = entry["screen_buffer"]
                cv2.imwrite(file_path + ".png", screen_buffer)

    def add_plane(self, a, b, c, d, pcl_size):
        entry = {
            "entry_type": "plane",
            "time": time.time(),
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "pcl_size": pcl_size}
        self.entries.append(entry)

    def add_filtered(self, filtered):
        entry = {
            "entry_type": "filtered",
            "time": time.time(),
            "image": filtered}
        self.entries.append(entry)

    def add_prev_rect_right(self, prev_rect_right):
        entry = {
            "entry_type": "prev_rect_right",
            "time": time.time(),
            "image": prev_rect_right}
        self.entries.append(entry)

    def add_preview_bb_free(self, preview_bb_free):
        entry = {
            "entry_type": "preview_bb_free",
            "time": time.time(),
            "image": preview_bb_free}
        self.entries.append(entry)

    def add_screen_buffer(self, screen_buffer):
        #info = np.iinfo(screen_buffer.dtype)  # Get the information of the incoming image type
        #screen_buffer = screen_buffer.astype(np.float64) / info.max  # normalize the data to 0 - 1
        #screen_buffer = 255 * screen_buffer  # Now scale by 255
        #screen_buffer = screen_buffer.astype(np.uint8)
        max = np.max(screen_buffer)
        min = np.min(screen_buffer)
        screen_buffer *= 255  # or any coefficient
        screen_buffer = screen_buffer.astype(np.uint8)
        entry = {
            "entry_type": "screen_buffer",
            "time": time.time(),
            "image": screen_buffer}
        self.entries.append(entry)