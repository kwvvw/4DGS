#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import random
import json
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset_readers import read_timestamps, readColmapCameras

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.args = args  # 保存为实例变量
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # ========== 使用带时间戳的 COLMAP 加载 ==========
        timestamps_dict = read_timestamps(args.source_path)

        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
        if os.path.exists(os.path.join(args.source_path, "sparse", "0", "cameras.bin")):
            cam_extrinsics = read_extrinsics_binary(os.path.join(args.source_path, "sparse", "0", "images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(args.source_path, "sparse", "0", "cameras.bin"))
        else:
            cam_extrinsics = read_extrinsics_text(os.path.join(args.source_path, "sparse", "0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(args.source_path, "sparse", "0", "cameras.txt"))

        images_folder = os.path.join(args.source_path, args.images) if args.images else os.path.join(args.source_path, "images")
        all_cameras = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, timestamps_dict)

        # 划分训练/测试集
        if args.eval:
            all_cameras.sort(key=lambda x: x.image_name)
            test_indices = list(range(0, len(all_cameras), 8))
            train_cameras = [c for i, c in enumerate(all_cameras) if i not in test_indices]
            test_cameras = [c for i, c in enumerate(all_cameras) if i in test_indices]
        else:
            train_cameras = all_cameras
            test_cameras = []

        # 设置 is_test 标志
        for cam in train_cameras:
            cam.is_test = False
        for cam in test_cameras:
            cam.is_test = True

        # 构建 scene_info
        class SceneInfo:
            pass
        scene_info = SceneInfo()
        scene_info.train_cameras = train_cameras
        scene_info.test_cameras = test_cameras
        scene_info.nerf_normalization = {"radius": 1.0}
        scene_info.point_cloud = None

        # 计算场景半径
        if train_cameras:
            centers = torch.stack([cam.camera_center for cam in train_cameras])
            radius = torch.max(torch.norm(centers, dim=-1)).item()
            scene_info.nerf_normalization["radius"] = radius
        else:
            radius = 1.0

        # 加载点云
        if not self.loaded_iter:
            points3d_path = os.path.join(args.source_path, "sparse", "0", "points3D.bin")
            if not os.path.exists(points3d_path):
                points3d_path = os.path.join(args.source_path, "sparse", "0", "points3D.txt")
            if os.path.exists(points3d_path):
                from scene.colmap_loader import read_points3D_binary, read_points3D_text
                if points3d_path.endswith(".bin"):
                    xyz, rgb, _ = read_points3D_binary(points3d_path)
                else:
                    xyz, rgb, _ = read_points3D_text(points3d_path)
                from scene.gaussian_model import BasicPointCloud
                scene_info.point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=None)
            else:
                if train_cameras:
                    centers_np = centers.cpu().numpy()
                    scene_info.point_cloud = BasicPointCloud(points=centers_np, colors=np.zeros((len(centers_np), 3)), normals=None)
                else:
                    scene_info.point_cloud = BasicPointCloud(points=np.zeros((1, 3)), colors=np.zeros((1, 3)), normals=None)

        # 跳过 input.ply 保存（避免 ply_path 缺失）
        if not self.loaded_iter:
            pass

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, getattr(scene_info, 'is_nerf_synthetic', False), False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, getattr(scene_info, 'is_nerf_synthetic', False), True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]