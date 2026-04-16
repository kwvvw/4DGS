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

import argparse
from argparse import ArgumentParser, Namespace
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0]), action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0]), type=t, default=value)
            else:
                if t == bool:
                    group.add_argument("--" + key, action="store_true", default=value)
                else:
                    group.add_argument("--" + key, type=t, default=value)

    def extract(self, args):
        group = GroupParams()
        for attr, value in vars(self).items():
            if attr.startswith("_"):
                attr = attr[1:]
            setattr(group, attr, getattr(args, attr))
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        # ====== 新增：傅里叶级数参数 ======
        self.fourier_degree = 4          # 傅里叶级数阶数 L
        self.base_freq = 1.0             # 基频（对应归一化时间周期为1）
        # ================================

        # ====== 新增：曝光训练开关 ======
        self.train_test_exp = False      # 是否训练每张图像的曝光补偿
        # ==============================

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.model_path = os.path.abspath(g.model_path)
        g.images = os.path.abspath(g.images) if g.images != "images" else g.images
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False  # 新增：是否启用抗锯齿
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        # 深度正则化参数
        self.depth_l1_weight_init = 0.0
        self.depth_l1_weight_final = 0.0

        # 曝光补偿参数
        self.exposure_lr_init = 0.0
        self.exposure_lr_final = 0.0
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        # ❌ 删除这一行： self.train_test_exp = False
        # 该参数已移至 ModelParams，避免重复定义

        # 优化器类型
        self.optimizer_type = "default"

        # ====== 新增：时间平滑正则化参数 ======
        self.time_smooth_weight = 0.001          # 时间平滑损失权重
        self.time_smooth_start_iter = 1000       # 开始施加正则化的迭代次数
        # ===================================

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmline = argparse.Namespace()
    for group in parser._action_groups:
        for a in group._group_actions:
            if a.dest != "help" and not hasattr(cmline, a.dest):
                setattr(cmline, a.dest, a.default)
    for key, value in vars(cmline).items():
        if value == "==SUPPRESS==":
            setattr(cmline, key, None)
    return cmline