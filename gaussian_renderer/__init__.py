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

import math
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           use_trained_exp=False, separate_sh=False):
    """
    从给定视点渲染场景，支持动态高斯（根据相机时间计算属性）。
    """
    # 背景颜色处理
    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")
    elif bg_color.dim() == 1:
        bg_color = bg_color.unsqueeze(0)  # (1,3)

    # 获取当前相机时间（归一化，默认静态为0.0）
    t = getattr(viewpoint_camera, 'time', 0.0)

    # ----- 动态属性获取（带安全回退）-----
    # 位置
    if hasattr(pc, 'get_xyz_at_time'):
        means3D = pc.get_xyz_at_time(t)
    else:
        means3D = pc.get_xyz  # 静态版本
    means3D = means3D.contiguous()

    # 缩放
    if hasattr(pc, 'get_scaling_at_time'):
        scales = pc.get_scaling_at_time(t)
    else:
        scales = pc.get_scaling
    scales = scales.contiguous()

    # 旋转
    if hasattr(pc, 'get_rotation_at_time'):
        rotations = pc.get_rotation_at_time(t)
    else:
        rotations = pc.get_rotation
    rotations = rotations.contiguous()

    # 不透明度
    if hasattr(pc, 'get_opacity_at_time'):
        opacities = pc.get_opacity_at_time(t)
    else:
        opacities = pc.get_opacity
    opacities = opacities.contiguous()

    # 球谐特征
    if hasattr(pc, 'get_features_at_time'):
        features_dc, features_rest = pc.get_features_at_time(t)
    else:
        features_dc, features_rest = pc.get_features
    if features_rest.shape[1] > 0:
        shs = torch.cat([features_dc, features_rest], dim=1)
    else:
        shs = features_dc

    # ----- 计算屏幕空间投影坐标 means2D -----
    # 将 3D 点转换为齐次坐标
    ones = torch.ones_like(means3D[:, :1])
    means3D_hom = torch.cat([means3D, ones], dim=1)  # (N, 4)

    # 应用世界-视图-投影变换
    means_clip = means3D_hom @ viewpoint_camera.world_view_transform  # (N, 4)
    means_clip = means_clip @ viewpoint_camera.projection_matrix       # (N, 4)

    # 透视除法得到 NDC 坐标
    means_ndc = means_clip[:, :3] / means_clip[:, 3:4]  # (N, 3)

    # 转换为屏幕像素坐标 (原点在左上角)
    width = viewpoint_camera.image_width
    height = viewpoint_camera.image_height
    means2D = torch.zeros_like(means_ndc[:, :2])
    means2D[:, 0] = (means_ndc[:, 0] + 1.0) * 0.5 * width
    means2D[:, 1] = (1.0 - means_ndc[:, 1]) * 0.5 * height  # Y 轴翻转

    # 关键修复：将坐标限制在有效屏幕范围内，防止光栅化器分配异常巨大的缓冲区
    means2D[:, 0].clamp_(min=0.0, max=width - 1.0)
    means2D[:, 1].clamp_(min=0.0, max=height - 1.0)

    # 可选：打印调试信息（若显存仍异常，可取消注释观察）
    # print(f"Image size: {width}x{height}, means2D range: x=[{means2D[:,0].min():.1f}, {means2D[:,0].max():.1f}], y=[{means2D[:,1].min():.1f}, {means2D[:,1].max():.1f}]")

    # ----- 设置光栅化参数 -----
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color.squeeze(0),  # (3,) 形状
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False          # 满足新版 rasterizer 要求
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 清理显存碎片（可选）
    torch.cuda.empty_cache()

    # ----- 执行渲染 (传入 means2D) -----
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,            # 必须提供
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )

    # 曝光补偿
    if use_trained_exp and hasattr(viewpoint_camera, 'exposure_a') and hasattr(viewpoint_camera, 'exposure_b'):
        exposure_a = viewpoint_camera.exposure_a
        exposure_b = viewpoint_camera.exposure_b
        rendered_image = rendered_image * exposure_a + exposure_b

    # 返回结果包
    render_pkg = {
        "render": rendered_image,
        "viewspace_points": None,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth
    }
    return render_pkg