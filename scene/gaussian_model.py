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
# 扩展：时间动态 3D 高斯模型 (傅里叶级数)
# 作者：基于原始 Gaussian Splatting 实现，添加时间维度建模
#

import torch
import torch.nn as nn
import numpy as np

# ---------- 辅助函数 (从原项目内联，保证独立性) ----------
def build_rotation(q):
    """四元数转旋转矩阵 (batch)"""
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device=q.device)
    rot[:, 0, 0] = 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    rot[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    rot[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    rot[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    rot[:, 1, 1] = 1 - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2)
    rot[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    rot[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    rot[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    rot[:, 2, 2] = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return rot

def strip_symmetric(symmat):
    """保留对称矩阵的上三角部分，使梯度传播稳定"""
    return symmat

def build_scaling_rotation(s, r):
    """从缩放和旋转构建协方差矩阵的 L 矩阵"""
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L

def RGB2SH(rgb):
    """RGB 颜色转球谐基函数 (直流分量)"""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class BasicPointCloud:
    """点云数据结构 (简化)"""
    def __init__(self, points, colors, normals=None):
        self.points = points
        self.colors = colors
        self.normals = normals

# ---------- 时间动态高斯模型 ----------
class GaussianModel:
    def __init__(self, sh_degree: int, optimizer_type: str = "default",
                 fourier_degree: int = 4, base_freq: float = 1.0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.optimizer_type = optimizer_type

        # 傅里叶参数
        self.fourier_degree = fourier_degree      # 阶数 L
        self.base_freq = base_freq                # 基频 (通常为 1.0)

        # 静态参数 (直流分量)
        self._xyz = torch.empty(0)                # 位置直流 (N,3)
        self._features_dc = torch.empty(0)        # 球谐直流 (N,3,1)
        self._features_rest = torch.empty(0)      # 球谐高阶 (N,3,(max_sh+1)^2-1)
        self._scaling = torch.empty(0)            # 缩放对数直流 (N,3)
        self._rotation = torch.empty(0)           # 四元数直流 (N,4)
        self._opacity = torch.empty(0)            # 不透明度 logit 直流 (N,1)

        # 傅里叶交流系数 (sin/cos)
        self._xyz_sin = torch.empty(0)            # (N, L, 3)
        self._xyz_cos = torch.empty(0)
        self._scaling_sin = torch.empty(0)        # (N, L, 3)
        self._scaling_cos = torch.empty(0)
        self._rotation_sin = torch.empty(0)       # (N, L, 4)
        self._rotation_cos = torch.empty(0)
        self._opacity_sin = torch.empty(0)        # (N, L, 1)
        self._opacity_cos = torch.empty(0)

        # 辅助变量
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.exposure_optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 0
        self.training_args = None                 # 保存训练配置

        self.setup_functions()

    def setup_functions(self):
        """激活函数与协方差构建"""
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize

    def create_fourier_params(self, num_points):
        """初始化傅里叶系数为零"""
        self._xyz_sin = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 3))
        self._xyz_cos = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 3))
        self._scaling_sin = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 3))
        self._scaling_cos = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 3))
        self._rotation_sin = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 4))
        self._rotation_cos = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 4))
        self._opacity_sin = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 1))
        self._opacity_cos = nn.Parameter(torch.zeros(num_points, self.fourier_degree, 1))

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """从点云初始化模型"""
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.cdist(fused_point_cloud, fused_point_cloud).mean(dim=1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.create_fourier_params(self._xyz.shape[0])

    def training_setup(self, training_args):
        """配置优化器"""
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        # 学习率
        self.position_lr_init = training_args.position_lr_init
        self.position_lr_final = training_args.position_lr_final
        self.position_lr_delay_mult = training_args.position_lr_delay_mult
        self.position_lr_max_steps = training_args.position_lr_max_steps

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._xyz_sin, self._xyz_cos],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.1, "name": "xyz_fourier"},
            {'params': [self._scaling_sin, self._scaling_cos],
             'lr': training_args.scaling_lr * 0.1, "name": "scaling_fourier"},
            {'params': [self._rotation_sin, self._rotation_cos],
             'lr': training_args.rotation_lr * 0.1, "name": "rotation_fourier"},
            {'params': [self._opacity_sin, self._opacity_cos],
             'lr': training_args.opacity_lr * 0.1, "name": "opacity_fourier"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = None
        if hasattr(training_args, 'train_test_exp') and training_args.train_test_exp:
            # 曝光优化器 (原项目保留接口，此处略)
            pass

    def update_learning_rate(self, iteration):
        """学习率指数衰减 (仅位置)"""
        if self.position_lr_delay_mult < 1.0:
            # 延迟衰减
            if iteration < self.position_lr_max_steps:
                lr = self.position_lr_init * self.spatial_lr_scale
            else:
                lr = self.position_lr_final * self.spatial_lr_scale
        else:
            # 指数衰减
            progress = iteration / self.position_lr_max_steps
            lr = self.position_lr_init * (self.position_lr_final / self.position_lr_init) ** progress
            lr *= self.spatial_lr_scale

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr
            # 傅里叶位置系数学习率随同缩放
            elif param_group["name"] == "xyz_fourier":
                param_group['lr'] = lr * 0.1

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _fourier_evaluate(self, dc, sin_coeff, cos_coeff, t):
        """
        通用傅里叶求值: dc + Σ [ sin(iωt)*A_i + cos(iωt)*B_i ]
        dc: (N, D)
        sin_coeff, cos_coeff: (N, L, D)
        t: 可以是标量 float 或 (B,) 的 torch.Tensor
        返回: 若 t 为标量返回 (N, D)；若 t 为张量返回 (B, N, D)
        """
        # 确保所有张量在同一设备
        device = dc.device
        sin_coeff = sin_coeff.to(device)
        cos_coeff = cos_coeff.to(device)

        # 将 t 转换为张量并移至相同设备
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        else:
            t = t.to(device)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        omega = 2 * torch.pi * self.base_freq * t  # (B,)
        B = omega.shape[0]
        N, D = dc.shape[0], dc.shape[1]

        result = dc.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        for i in range(1, self.fourier_degree + 1):
            angle = i * omega.view(B, 1, 1)  # (B, 1, 1)
            sin_term = torch.sin(angle) * sin_coeff[:, i - 1, :].unsqueeze(0)
            cos_term = torch.cos(angle) * cos_coeff[:, i - 1, :].unsqueeze(0)
            result = result + sin_term + cos_term

        if B == 1 and t.numel() == 1:
            result = result.squeeze(0)
        return result

    def _fourier_derivative(self, sin_coeff, cos_coeff, t):
        """
        傅里叶级数对时间的导数 (直流部分导数为0)
        sin_coeff, cos_coeff: (N, L, D)
        t: 标量 float 或 (B,) 张量
        返回: 若 t 为标量返回 (N, D)；若 t 为张量返回 (B, N, D)
        """
        device = sin_coeff.device
        sin_coeff = sin_coeff.to(device)
        cos_coeff = cos_coeff.to(device)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        else:
            t = t.to(device)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        omega = 2 * torch.pi * self.base_freq * t  # (B,)
        B = omega.shape[0]
        N = sin_coeff.shape[0]
        D = sin_coeff.shape[2]

        deriv = torch.zeros(B, N, D, device=device)
        for i in range(1, self.fourier_degree + 1):
            angle = i * omega.view(B, 1, 1)
            coeff = 2 * torch.pi * self.base_freq * i
            term = coeff * (torch.cos(angle) * sin_coeff[:, i - 1, :].unsqueeze(0) -
                            torch.sin(angle) * cos_coeff[:, i - 1, :].unsqueeze(0))
            deriv += term

        if B == 1 and t.numel() == 1:
            deriv = deriv.squeeze(0)
        return deriv

    # 动态属性获取
    def get_xyz_at_time(self, t):
        return self._fourier_evaluate(self._xyz, self._xyz_sin, self._xyz_cos, t)

    def get_scaling_at_time(self, t):
        scaling_log = self._fourier_evaluate(self._scaling, self._scaling_sin, self._scaling_cos, t)
        return torch.exp(scaling_log)

    def get_rotation_at_time(self, t):
        quat = self._fourier_evaluate(self._rotation, self._rotation_sin, self._rotation_cos, t)
        return self.rotation_activation(quat)

    def get_opacity_at_time(self, t):
        raw = self._fourier_evaluate(self._opacity, self._opacity_sin, self._opacity_cos, t)
        return self.opacity_activation(raw)

    def get_features_at_time(self, t):
        # 球谐特征暂不随时间变化 (可根据需要扩展)
        return self._features_dc, self._features_rest

    # 导数
    def get_xyz_derivative_at_time(self, t):
        return self._fourier_derivative(self._xyz_sin, self._xyz_cos, t)

    def get_scaling_derivative_at_time(self, t):
        return self._fourier_derivative(self._scaling_sin, self._scaling_cos, t)

    def get_rotation_derivative_at_time(self, t):
        return self._fourier_derivative(self._rotation_sin, self._rotation_cos, t)

    def get_opacity_derivative_at_time(self, t):
        return self._fourier_derivative(self._opacity_sin, self._opacity_cos, t)

    # ---------- 密度控制 (同步傅里叶系数) ----------
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_scaling, new_rotation, new_opacity,
                              new_xyz_sin, new_xyz_cos,
                              new_scaling_sin, new_scaling_cos,
                              new_rotation_sin, new_rotation_cos,
                              new_opacity_sin, new_opacity_cos):
        """替换所有参数 (用于 densify 后)"""
        self._xyz = nn.Parameter(new_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(new_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(new_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(new_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(new_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(new_opacity.requires_grad_(True))
        self._xyz_sin = nn.Parameter(new_xyz_sin.requires_grad_(True))
        self._xyz_cos = nn.Parameter(new_xyz_cos.requires_grad_(True))
        self._scaling_sin = nn.Parameter(new_scaling_sin.requires_grad_(True))
        self._scaling_cos = nn.Parameter(new_scaling_cos.requires_grad_(True))
        self._rotation_sin = nn.Parameter(new_rotation_sin.requires_grad_(True))
        self._rotation_cos = nn.Parameter(new_rotation_cos.requires_grad_(True))
        self._opacity_sin = nn.Parameter(new_opacity_sin.requires_grad_(True))
        self._opacity_cos = nn.Parameter(new_opacity_cos.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.optimizer = None
        self.training_setup(self.training_args)

    def prune_points(self, mask):
        """删除高斯点 (mask=True 的点被删除)"""
        valid_mask = ~mask
        keep_idx = torch.where(valid_mask)[0]

        self._xyz = nn.Parameter(self._xyz[keep_idx].requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc[keep_idx].requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest[keep_idx].requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[keep_idx].requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[keep_idx].requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[keep_idx].requires_grad_(True))

        self._xyz_sin = nn.Parameter(self._xyz_sin[keep_idx].requires_grad_(True))
        self._xyz_cos = nn.Parameter(self._xyz_cos[keep_idx].requires_grad_(True))
        self._scaling_sin = nn.Parameter(self._scaling_sin[keep_idx].requires_grad_(True))
        self._scaling_cos = nn.Parameter(self._scaling_cos[keep_idx].requires_grad_(True))
        self._rotation_sin = nn.Parameter(self._rotation_sin[keep_idx].requires_grad_(True))
        self._rotation_cos = nn.Parameter(self._rotation_cos[keep_idx].requires_grad_(True))
        self._opacity_sin = nn.Parameter(self._opacity_sin[keep_idx].requires_grad_(True))
        self._opacity_cos = nn.Parameter(self._opacity_cos[keep_idx].requires_grad_(True))

        self.max_radii2D = self.max_radii2D[keep_idx]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_idx]
        self.denom = self.denom[keep_idx]

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """克隆梯度大的高斯点 (复制所有参数，包括傅里叶系数)"""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_at_time(0.0), dim=1).values <=
                                              self.percent_dense * scene_extent)
        if not selected_pts_mask.any():
            return

        # 复制原始数据
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        new_xyz_sin = self._xyz_sin[selected_pts_mask]
        new_xyz_cos = self._xyz_cos[selected_pts_mask]
        new_scaling_sin = self._scaling_sin[selected_pts_mask]
        new_scaling_cos = self._scaling_cos[selected_pts_mask]
        new_rotation_sin = self._rotation_sin[selected_pts_mask]
        new_rotation_cos = self._rotation_cos[selected_pts_mask]
        new_opacity_sin = self._opacity_sin[selected_pts_mask]
        new_opacity_cos = self._opacity_cos[selected_pts_mask]

        # 拼接
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0).requires_grad_(True))

        self._xyz_sin = nn.Parameter(torch.cat([self._xyz_sin, new_xyz_sin], dim=0).requires_grad_(True))
        self._xyz_cos = nn.Parameter(torch.cat([self._xyz_cos, new_xyz_cos], dim=0).requires_grad_(True))
        self._scaling_sin = nn.Parameter(torch.cat([self._scaling_sin, new_scaling_sin], dim=0).requires_grad_(True))
        self._scaling_cos = nn.Parameter(torch.cat([self._scaling_cos, new_scaling_cos], dim=0).requires_grad_(True))
        self._rotation_sin = nn.Parameter(torch.cat([self._rotation_sin, new_rotation_sin], dim=0).requires_grad_(True))
        self._rotation_cos = nn.Parameter(torch.cat([self._rotation_cos, new_rotation_cos], dim=0).requires_grad_(True))
        self._opacity_sin = nn.Parameter(torch.cat([self._opacity_sin, new_opacity_sin], dim=0).requires_grad_(True))
        self._opacity_cos = nn.Parameter(torch.cat([self._opacity_cos, new_opacity_cos], dim=0).requires_grad_(True))

        # 重置辅助变量并重建优化器
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.optimizer = None
        self.training_setup(self.training_args)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """分裂大尺度高斯点，新点继承傅里叶系数"""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling_at_time(0.0), dim=1).values >
                                              self.percent_dense * scene_extent)
        if not selected_pts_mask.any():
            return

        # 收集待分裂点的索引
        pts_idx = torch.where(selected_pts_mask)[0]
        # 对每个点生成 N 个新点
        new_xyz_list = []
        new_scaling_list = []
        new_rotation_list = []
        new_opacity_list = []
        new_features_dc_list = []
        new_features_rest_list = []
        new_xyz_sin_list = []
        new_xyz_cos_list = []
        new_scaling_sin_list = []
        new_scaling_cos_list = []
        new_rotation_sin_list = []
        new_rotation_cos_list = []
        new_opacity_sin_list = []
        new_opacity_cos_list = []

        for idx in pts_idx:
            # 原参数
            scale = torch.exp(self._scaling[idx])
            scale_log = self._scaling[idx]
            rot = self._rotation[idx]
            pos = self._xyz[idx]
            op = self._opacity[idx]
            feat_dc = self._features_dc[idx]
            feat_rest = self._features_rest[idx]
            # 傅里叶系数
            xyz_sin = self._xyz_sin[idx]
            xyz_cos = self._xyz_cos[idx]
            scale_sin = self._scaling_sin[idx]
            scale_cos = self._scaling_cos[idx]
            rot_sin = self._rotation_sin[idx]
            rot_cos = self._rotation_cos[idx]
            op_sin = self._opacity_sin[idx]
            op_cos = self._opacity_cos[idx]

            # 生成随机方向
            stds = scale[None, :].repeat(N, 1)  # (N,3)
            samples = torch.randn(N, 3, device=scale.device) * stds
            rots_mat = build_rotation(rot[None, :])  # (1,3,3)
            offsets = (samples @ rots_mat.squeeze(0).T)  # (N,3)
            new_pos = pos[None, :] + offsets
            new_scale_log = scale_log[None, :] + torch.log(0.5 * torch.ones(N, 3, device=scale.device))
            new_rot = rot[None, :].repeat(N, 1)
            new_op = op[None, :].repeat(N, 1)
            new_feat_dc = feat_dc[None, :].repeat(N, 1)
            new_feat_rest = feat_rest[None, :].repeat(N, 1)

            # 傅里叶系数直接复制 (因为时间动态模式应相同)
            new_xyz_sin = xyz_sin[None, :, :].repeat(N, 1, 1)
            new_xyz_cos = xyz_cos[None, :, :].repeat(N, 1, 1)
            new_scale_sin = scale_sin[None, :, :].repeat(N, 1, 1)
            new_scale_cos = scale_cos[None, :, :].repeat(N, 1, 1)
            new_rot_sin = rot_sin[None, :, :].repeat(N, 1, 1)
            new_rot_cos = rot_cos[None, :, :].repeat(N, 1, 1)
            new_op_sin = op_sin[None, :, :].repeat(N, 1, 1)
            new_op_cos = op_cos[None, :, :].repeat(N, 1, 1)

            new_xyz_list.append(new_pos)
            new_scaling_list.append(new_scale_log)
            new_rotation_list.append(new_rot)
            new_opacity_list.append(new_op)
            new_features_dc_list.append(new_feat_dc)
            new_features_rest_list.append(new_feat_rest)
            new_xyz_sin_list.append(new_xyz_sin)
            new_xyz_cos_list.append(new_xyz_cos)
            new_scaling_sin_list.append(new_scale_sin)
            new_scaling_cos_list.append(new_scale_cos)
            new_rotation_sin_list.append(new_rot_sin)
            new_rotation_cos_list.append(new_rot_cos)
            new_opacity_sin_list.append(new_op_sin)
            new_opacity_cos_list.append(new_op_cos)

        # 拼接所有新点
        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_scaling = torch.cat(new_scaling_list, dim=0)
        new_rotation = torch.cat(new_rotation_list, dim=0)
        new_opacity = torch.cat(new_opacity_list, dim=0)
        new_features_dc = torch.cat(new_features_dc_list, dim=0)
        new_features_rest = torch.cat(new_features_rest_list, dim=0)
        new_xyz_sin = torch.cat(new_xyz_sin_list, dim=0)
        new_xyz_cos = torch.cat(new_xyz_cos_list, dim=0)
        new_scaling_sin = torch.cat(new_scaling_sin_list, dim=0)
        new_scaling_cos = torch.cat(new_scaling_cos_list, dim=0)
        new_rotation_sin = torch.cat(new_rotation_sin_list, dim=0)
        new_rotation_cos = torch.cat(new_rotation_cos_list, dim=0)
        new_opacity_sin = torch.cat(new_opacity_sin_list, dim=0)
        new_opacity_cos = torch.cat(new_opacity_cos_list, dim=0)

        # 保留未分裂的点，并删除原分裂点
        keep_mask = ~selected_pts_mask
        self._xyz = nn.Parameter(torch.cat([self._xyz[keep_mask], new_xyz], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc[keep_mask], new_features_dc], dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest[keep_mask], new_features_rest], dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([self._scaling[keep_mask], new_scaling], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([self._rotation[keep_mask], new_rotation], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([self._opacity[keep_mask], new_opacity], dim=0).requires_grad_(True))

        self._xyz_sin = nn.Parameter(torch.cat([self._xyz_sin[keep_mask], new_xyz_sin], dim=0).requires_grad_(True))
        self._xyz_cos = nn.Parameter(torch.cat([self._xyz_cos[keep_mask], new_xyz_cos], dim=0).requires_grad_(True))
        self._scaling_sin = nn.Parameter(torch.cat([self._scaling_sin[keep_mask], new_scaling_sin], dim=0).requires_grad_(True))
        self._scaling_cos = nn.Parameter(torch.cat([self._scaling_cos[keep_mask], new_scaling_cos], dim=0).requires_grad_(True))
        self._rotation_sin = nn.Parameter(torch.cat([self._rotation_sin[keep_mask], new_rotation_sin], dim=0).requires_grad_(True))
        self._rotation_cos = nn.Parameter(torch.cat([self._rotation_cos[keep_mask], new_rotation_cos], dim=0).requires_grad_(True))
        self._opacity_sin = nn.Parameter(torch.cat([self._opacity_sin[keep_mask], new_opacity_sin], dim=0).requires_grad_(True))
        self._opacity_cos = nn.Parameter(torch.cat([self._opacity_cos[keep_mask], new_opacity_cos], dim=0).requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.optimizer = None
        self.training_setup(self.training_args)

    def reset_opacity(self):
        """重置不透明度 (用于防止死点)"""
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity_at_time(0.0), torch.ones_like(self.get_opacity_at_time(0.0)) * 0.01)
        )
        self._opacity = nn.Parameter(opacities_new.requires_grad_(True))

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """累积梯度统计 (用于密度控制)"""
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # ---------- 保存与恢复 ----------
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer else None,
            self.exposure_optimizer.state_dict() if self.exposure_optimizer else None,
            self._xyz_sin,
            self._xyz_cos,
            self._scaling_sin,
            self._scaling_cos,
            self._rotation_sin,
            self._rotation_cos,
            self._opacity_sin,
            self._opacity_cos,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         self.xyz_gradient_accum,
         self.denom,
         opt_dict,
         exp_opt_dict,
         self._xyz_sin,
         self._xyz_cos,
         self._scaling_sin,
         self._scaling_cos,
         self._rotation_sin,
         self._rotation_cos,
         self._opacity_sin,
         self._opacity_cos) = model_args
        self.training_setup(training_args)
        if opt_dict:
            self.optimizer.load_state_dict(opt_dict)
        if exp_opt_dict and self.exposure_optimizer:
            self.exposure_optimizer.load_state_dict(exp_opt_dict)

    # 兼容原接口的属性
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features_dc, self._features_rest

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)