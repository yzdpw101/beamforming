"""流形空间映射 (MSM/OMM) — 替代模型 Rs(x) = Yf + S·(Yc - Yc_ref)。"""

import numpy as np
from .base import compute_fine_psll


def run_manifold_mapping(Xc_star, coarse_model_func, n_vars, hfss_func,
                         max_iter=10, target_fine_psll=-30.0,
                         cma_iter=300, sigma=0.3, seed=0,
                         verbose=True, on_iter_end=None):
    """流形空间映射 (MSM/OMM) 主循环 (预留)。

    Args:
        Xc_star: 粗模型最优变量
        coarse_model_func: f(x) -> 1d dB 方向图
        n_vars: 优化变量维度
        hfss_func: 细模型函数, 返回 [Yf1, Yf2, ...]
        max_iter: 最大 MSM 迭代次数
        target_fine_psll: 目标细模型 PSLL
        cma_iter: CMA-ES 迭代数
        sigma: 初始步长
        seed: 随机种子
        verbose: 是否打印日志
        on_iter_end: 回调

    Returns:
        (X_best, best_psll, history)
    """
    raise NotImplementedError("MSM 主循环待实现")
