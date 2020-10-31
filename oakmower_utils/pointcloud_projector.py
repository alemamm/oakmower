"""
Author: Jan Lukas Augustin <jan.lukas.augustin@gmail.com>
Class for 3D projection of depth data, plane segmentation, visualization and potentially persisting point cloud images.
Check https://github.com/luxonis/depthai-experiments/tree/master/point-cloud-projection for a stand-alone example
License: GNU General Public License v3.0
"""

import open3d as o3d
import numpy as np
import os
import time


class PointCloudProjector():
    def __init__(self, intrinsic_file, visualize=True, persister=None, verbose=False, show_full_cloud=False):
        self.persister = persister
        self.depth_map = None
        self.rgb = None
        self.pcl = None
        self.pcl_combined = None
        self.visualize = visualize
        self.verbose = verbose
        self.show_full_cloud = show_full_cloud
        assert os.path.isfile(intrinsic_file), ("Intrisic file not found. Rerun the calibrate.py to generate intrinsic file")
            # print()
        self.pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_file)
        if self.visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Plane Segmentation")
            self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            a, b, c, d, pcl_size = 0, 0, 0, 0, 0
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10)
            [a, b, c, d] = plane_model
            inlier_cloud = pcd.select_by_index(inliers)
            pcl_size = np.asarray(inlier_cloud.points).size
            if self.show_full_cloud:
                inlier_cloud.paint_uniform_color([0.0, 1.0, 0])
                outlier_cloud = pcd.select_by_index(inliers, invert=True)
                outlier_cloud.paint_uniform_color([0.9, 0.1, 0.1])
                combined_cloud = inlier_cloud
                combined_cloud += outlier_cloud
                self.pcl.points = combined_cloud.points
                self.pcl.colors = combined_cloud.colors
            else:
                combined_cloud = inlier_cloud
                self.pcl.points = combined_cloud.points
            if self.verbose:
                print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                print("Number of poins in plane:", pcl_size)
        if self.persister is not None:
            self.persister.add_plane(a, b, c, d, pcl_size)
        if self.visualize:
            self.visualize_pcd()
        return {"a": a, "b": b, "c": c, "d": d, "pcl_size": pcl_size}

    def visualize_pcd(self):
        assert self.visualize , ("visualize is set False. Set visualize to True to see point cloud visualization")
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            #if self.persister is not None:
                #screen_buffer = self.vis.capture_screen_float_buffer(do_render=True)
                #self.persister.add_screen_buffer(np.asarray(screen_buffer))
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()
