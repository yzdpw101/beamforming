"""平面阵位置映射器 — 网格选通模式。

平面阵的位置优化与线阵不同：线阵用 Stick-Breaking 连续分配间距，
平面阵用网格候选点二值选通（thinning）。
"""

import numpy as np
from .base import BaseMapper


class PlanarMapper(BaseMapper):
    """平面阵网格选通映射器。

    在固定矩形网格上，通过 sigmoid 映射的控制变量决定每个候选阵元的激活状态。
    八邻域硬约束确保无相邻阵元。

    Args:
        Nx, Ny: x/y 方向网格点数
        dx, dy: x/y 方向间距（波长单位）
        dmin: 相邻阵元最小间距（波长单位, 用于碰撞检测）
    """

    def __init__(self, Nx: int, Ny: int, dx: float, dy: float, dmin: float = 0.5):
        super().__init__(Ne=Nx * Ny, L=max(Nx * dx, Ny * dy),
                         dmin=dmin)
        self._Nx = Nx
        self._Ny = Ny
        self._dx = dx
        self._dy = dy

        # 预生成候选网格
        xs = (np.arange(Nx) - (Nx - 1) / 2) * dx
        ys = (np.arange(Ny) - (Ny - 1) / 2) * dy
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)
        self._grid_x = self._grid_x.ravel()
        self._grid_y = self._grid_y.ravel()

    @property
    def n_vars(self) -> int:
        return self._Ne  # 每候选阵元一个选通变量

    def synthesize(self, opt_vector: np.ndarray) -> tuple:
        """将优化变量映射为激活阵元位置。

        Args:
            opt_vector: 选通变量 (Ncand,), sigmoid → 激活概率

        Returns:
            (x_active, y_active): 激活阵元坐标 tuple
        """
        v = np.asarray(opt_vector, dtype=float)
        # sigmoid 映射 → 激活概率
        prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        active = prob > 0.5

        # 八邻域碰撞检测
        active = self._collision_check(active)

        return self._grid_x[active].copy(), self._grid_y[active].copy()

    def _collision_check(self, active: np.ndarray) -> np.ndarray:
        """八邻域硬约束：无相邻阵元同时激活。"""
        active_2d = active.reshape(self._Ny, self._Nx)
        for i in range(self._Ny):
            for j in range(self._Nx):
                if not active_2d[i, j]:
                    continue
                # 检查八邻域
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self._Ny and 0 <= nj < self._Nx:
                            if active_2d[ni, nj]:
                                # 冲突：保留索引更小的
                                if ni < i or (ni == i and nj < j):
                                    active_2d[i, j] = False
                                else:
                                    active_2d[ni, nj] = False
        return active_2d.ravel()
