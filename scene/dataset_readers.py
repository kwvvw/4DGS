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
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
import glob
from pathlib import Path


# ====== 新增：时间戳读取函数 ======
def read_timestamps(source_path):
    """
    从 source_path/timestamps.txt 读取每张图像的时间戳（秒），并归一化到 [0,1]。
    文件格式：每行 "image_name time_in_seconds"
    返回字典 {image_name: normalized_time}
    """
    timestamps_file = os.path.join(source_path, "timestamps.txt")
    if not os.path.exists(timestamps_file):
        print(f"[Warning] No timestamps.txt found in {source_path}, setting all times to 0.0")
        return {}

    timestamps_sec = {}
    with open(timestamps_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"[Warning] Invalid line in timestamps.txt: {line}, skipping")
                continue
            img_name, t_sec = parts[0], float(parts[1])
            timestamps_sec[img_name] = t_sec

    if not timestamps_sec:
        return {}

    # 归一化到 [0, 1]
    max_t = max(timestamps_sec.values())
    min_t = min(timestamps_sec.values())
    if max_t == min_t:
        normalized = {img: 0.0 for img in timestamps_sec}
    else:
        normalized = {img: (t - min_t) / (max_t - min_t) for img, t in timestamps_sec.items()}

    return normalized


# =================================

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, timestamps_dict, eval_split=False):
    cameras = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        img_path = os.path.join(images_folder, extr.name)
        if not os.path.exists(img_path):
            print(f"[Warning] Image {img_path} not found, skipping.")
            continue

        image = Image.open(img_path)
        image = np.array(image).astype(np.float32) / 255.0
        if image.shape[2] == 4:
            image = image[..., :3]
        image = torch.from_numpy(image).permute(2, 0, 1)

        time = timestamps_dict.get(extr.name, 0.0)

        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        FovX = focal2fov(fx, image.shape[2])
        FovY = focal2fov(fy, image.shape[1])

        # 修正点：直接调用方法，不传参数
        R = np.transpose(extr.qvec2rotmat())
        T = extr.tvec

        camera = Camera(
            colmap_id=extr.camera_id,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            gt_alpha_mask=None,
            image_name=extr.name,
            uid=idx,
            time=time,
            data_device="cuda",
            image_path=img_path,
            depth_path="",  # 新增
            depth_params={}  # 新增
        )

        cameras.append(camera)
    return cameras

class Scene(Dataset):
    """
    场景类，负责加载 COLMAP 数据、图像，并生成训练/测试相机列表。
    修改后支持时间戳。
    """

    def __init__(self, args, model_path, load_iteration=-1, shuffle=True, resolution_scales=[1.0]):
        self.args = args
        self.model_path = model_path
        self.load_iteration = load_iteration
        self.shuffle = shuffle

        # 读取时间戳
        self.timestamps_dict = read_timestamps(args.source_path)

        # 读取 COLMAP 数据
        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, \
            read_intrinsics_text
        # 根据文件格式选择读取方式
        if os.path.exists(os.path.join(args.source_path, "sparse", "0", "cameras.bin")):
            cam_extrinsics = read_extrinsics_binary(os.path.join(args.source_path, "sparse", "0", "images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(args.source_path, "sparse", "0", "cameras.bin"))
        else:
            cam_extrinsics = read_extrinsics_text(os.path.join(args.source_path, "sparse", "0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(args.source_path, "sparse", "0", "cameras.txt"))

        # 获取图像文件夹路径
        images_folder = os.path.join(args.source_path, args.images) if args.images else os.path.join(args.source_path,
                                                                                                     "images")

        # 生成所有相机
        all_cameras = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, self.timestamps_dict)

        # 根据 eval 标志划分训练/测试集
        if args.eval:
            # 简单示例：按图像名排序后，每隔8张取一个作为测试
            all_cameras.sort(key=lambda x: x.image_name)
            test_indices = list(range(0, len(all_cameras), 8))
            self.train_cameras = [c for i, c in enumerate(all_cameras) if i not in test_indices]
            self.test_cameras = [c for i, c in enumerate(all_cameras) if i in test_indices]
        else:
            self.train_cameras = all_cameras
            self.test_cameras = []

        # 可选：分辨率缩放
        for scale in resolution_scales:
            if scale != 1.0:
                # 缩放图像和相机参数
                self._rescale_cameras(scale)

        print(f"Loaded {len(self.train_cameras)} training cameras, {len(self.test_cameras)} test cameras.")

    def _rescale_cameras(self, scale):
        """缩放图像分辨率，调整相机内参"""
        for cam in self.train_cameras + self.test_cameras:
            # 缩放图像
            orig_h, orig_w = cam.original_image.shape[1], cam.original_image.shape[2]
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            cam.original_image = torch.nn.functional.interpolate(cam.original_image.unsqueeze(0), size=(new_h, new_w),
                                                                 mode='bilinear').squeeze(0)
            cam.image_width = new_w
            cam.image_height = new_h
            # 调整 FOV（因为焦距等比例缩放）
            cam.FoVx = focal2fov(fov2focal(cam.FoVx, orig_w) * scale, new_w)
            cam.FoVy = focal2fov(fov2focal(cam.FoVy, orig_h) * scale, new_h)
            # 重新计算投影矩阵
            cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx,
                                                        fovY=cam.FoVy).transpose(0, 1).to(cam.data_device)
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras