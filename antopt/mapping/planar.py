"""平面阵位置映射器 — MGOM / ITSM2 / 网格选通。"""

import numpy as np
from .base import BaseMapper


# ═══════════════════════════════════════════════════
#  MGOM — 修正网格正交法
# ═══════════════════════════════════════════════════

class MGOM(BaseMapper):
    """修正网格正交法 (Modified Grid Orthogonal Method)。

    在矩形网格上通过打分选通 + 位置偏移实现稀布面阵映射。

    Args:
        Ne: 阵元总数（对称时 = 4 × Nq）
        Lx: x 方向孔径（波长单位）
        Ly: y 方向孔径（波长单位）
        dmin: 最小阵元间距（波长单位）
        is_symmetric: 是否关于原点四象限对称
    """

    def __init__(
        self,
        Ne: int,
        Lx: float,
        Ly: float,
        dmin: float,
        is_symmetric: bool = True,
    ):
        if Ne <= 0:
            raise ValueError(f"Ne 必须为正, 实际 {Ne}")
        if is_symmetric and Ne % 4 != 0:
            raise ValueError(f"对称阵列 Ne 必须为 4 的倍数, 实际 {Ne}")
        if Lx <= 0 or Ly <= 0:
            raise ValueError(f"Lx/Ly 必须 > 0, 实际 Lx={Lx} Ly={Ly}")
        if dmin <= 0:
            raise ValueError(f"dmin 必须 > 0, 实际 {dmin}")

        self._Lx = Lx
        self._Ly = Ly
        self._dmin = dmin
        self._is_sym = is_symmetric

        # 第一象限边界
        if is_symmetric:
            self._x0 = 0.5 * dmin
            self._y0 = 0.5 * dmin
            self._x1 = Lx / 2.0
            self._y1 = Ly / 2.0
            self._Nq = Ne // 4
        else:
            self._x0 = -Lx / 2.0
            self._y0 = -Ly / 2.0
            self._x1 = Lx / 2.0
            self._y1 = Ly / 2.0
            self._Nq = Ne

        self._Pm = int((self._y1 - self._y0) / dmin) + 1  # 行数
        self._Qm = int((self._x1 - self._x0) / dmin) + 1  # 列数

        super().__init__(
            Ne=Ne, L=Lx, dmin=dmin,
            is_symmetric=is_symmetric, is_fixed_aperture=True,
        )

    # ── 属性 ──

    @property
    def Lx(self) -> float:
        return self._Lx

    @property
    def Ly(self) -> float:
        return self._Ly

    @property
    def Nq(self) -> int:
        return self._Nq

    @property
    def max_rows(self) -> int:
        return self._Pm

    @property
    def max_cols(self) -> int:
        return self._Qm

    @property
    def max_cells(self) -> int:
        return self._Pm * self._Qm

    @property
    def n_vars(self) -> int:
        """变量维度 = 选通分 + x偏移 + y偏移。"""
        if self._is_sym:
            return (self.max_cells - 1) + 2 * (self.Nq - 1)
        return (self.max_cells - 4) + 2 * (self.Nq - 4)

    # ── 核心 ──

    def synthesize(self, opt_vector: np.ndarray, expand_to_4q: bool = True) -> tuple:
        """将优化变量映射为阵元坐标。

        Args:
            opt_vector: 优化变量, shape (n_vars,)
            expand_to_4q: 对称阵列是否展开到四象限

        Returns:
            (x, y): 阵元坐标, shape (Ne,) 或 (Nq,)
        """
        x = np.asarray(opt_vector, dtype=float)
        sel, ox, oy = self._parse(x)
        qx, qy = self._build(sel, ox, oy)

        if self._is_sym and expand_to_4q:
            return self._expand(qx, qy)
        return qx, qy

    # ── 内部 ──

    def _parse(self, x: np.ndarray) -> tuple:
        """解析变量向量 → (selection_mask, offset_x, offset_y)。"""
        Pm, Qm = self._Pm, self._Qm
        Nq = self._Nq

        if self._is_sym:
            # W: (max_cells-1,) 打分, 右下角固定=1
            n_w = self.max_cells - 1
            W = x[:n_w]
            idx = n_w
            # α: (Nq-1,) x偏移, 末尾固定=1
            alpha = np.empty(Nq)
            alpha[:-1] = x[idx:idx + Nq - 1]; alpha[-1] = 1.0
            idx += Nq - 1
            # β: (Nq-1,) y偏移, 末尾固定=1
            beta = np.empty(Nq)
            beta[:-1] = x[idx:idx + Nq - 1]; beta[-1] = 1.0
        else:
            # 非对称: 四个角固定
            n_w = self.max_cells - 4
            W = x[:n_w]
            idx = n_w
            alpha = np.empty(Nq)
            alpha[:Nq - 4] = x[idx:idx + Nq - 4]
            alp_fill = idx + Nq - 4
            idx = alp_fill
            beta = np.empty(Nq)
            beta[:Nq - 4] = x[idx:idx + Nq - 4]
            # 四个角固定
            alpha[-4:] = [0, 1, 0, 1]
            beta[-4:]  = [0, 0, 1, 1]

        # 选 top-Nq 高分单元
        sel = np.zeros(self.max_cells, dtype=bool)
        order = np.argsort(-W)  # 降序
        chosen = order[:Nq]
        sel[chosen] = True

        return sel, alpha, beta

    def _build(self, selection, alpha, beta) -> tuple:
        """构建第一象限坐标。"""
        Pm, Qm = self._Pm, self._Qm
        dmin = self._dmin
        x0, y0 = self._x0, self._y0
        x1, y1 = self._x1, self._y1

        sel = selection.reshape(Pm, Qm)
        X = np.zeros((Pm, Qm))
        Y = np.zeros((Pm, Qm))

        # 打平 alpha/beta 到选中的单元
        oX = np.zeros((Pm, Qm)); oY = np.zeros((Pm, Qm))
        k = 0
        for ri in range(Pm):
            for ci in range(Qm):
                if sel[ri, ci]:
                    oX[ri, ci] = alpha[k]
                    oY[ri, ci] = beta[k]
                    k += 1

        # ── Step 1: X 坐标 ──
        x_rem = x1 - x0 - (Qm - 1) * dmin  # 最后一列剩余空间
        for ri in range(Pm):
            for ci in range(Qm):
                if not sel[ri, ci]:
                    continue
                # 找同行右侧相邻选中单元
                ci_end = ci
                while ci_end < Qm - 1 and sel[ri, ci_end + 1]:
                    ci_end += 1

                if ci_end == Qm - 1:
                    # 延伸到最后列
                    if x_rem < 1e-12:
                        # 无剩余空间，均匀放置
                        X[ri, ci] = x0 + ci * dmin
                        for cj in range(ci + 1, Qm):
                            X[ri, cj] = X[ri, cj - 1] + dmin
                    else:
                        # 有剩余空间，用偏移分配
                        r = x_rem
                        X[ri, ci] = x0 + ci * dmin + oX[ri, ci] * r
                        r *= (1.0 - oX[ri, ci])
                        for cj in range(ci + 1, Qm):
                            X[ri, cj] = X[ri, cj - 1] + oX[ri, cj] * r + dmin
                            r *= (1.0 - oX[ri, cj])
                else:
                    # 未延伸到边缘，块内分配
                    n_in_block = ci_end - ci + 1
                    x_block = x0 + ci_end * dmin + dmin - (x0 + ci * dmin)
                    x_used = (n_in_block - 1) * dmin
                    r = x_block - x_used
                    X[ri, ci] = x0 + ci * dmin + oX[ri, ci] * r
                    r *= (1.0 - oX[ri, ci])
                    for cj in range(ci + 1, ci_end + 1):
                        X[ri, cj] = X[ri, cj - 1] + oX[ri, cj] * r + dmin
                        r *= (1.0 - oX[ri, cj])

                ci = ci_end  # 跳过已处理的块

        # ── Step 2: Y 坐标 ──
        y_rem = y1 - y0 - (Pm - 1) * dmin
        # 最后一行
        last = Pm - 1
        if y_rem < 1e-12:
            Y[last] = y1
        else:
            y_low = y0 + last * dmin
            for ci in range(Qm):
                if sel[last, ci]:
                    Y[last, ci] = np.clip(y_low + oY[last, ci] * y_rem, y_low, y1)

        # 从倒数第二行往上
        for ri in range(Pm - 2, -1, -1):
            y_low = y0 + ri * dmin
            y_up = y0 + (ri + 1) * dmin  # 默认上界

            for ci in range(Qm):
                if not sel[ri, ci]:
                    continue
                # 检查上一行 ±1 列的碰撞
                ub = y_up
                for dc in (-1, 0, 1):
                    cj = ci + dc
                    if 0 <= cj < Qm and sel[ri + 1, cj]:
                        dx = abs(X[ri, ci] - X[ri + 1, cj])
                        if dx < dmin:
                            dy_safe = np.sqrt(max(0, dmin * dmin - dx * dx))
                            if dy_safe >= dmin - 1e-12:
                                dy_safe = dmin
                            ub = min(ub, Y[ri + 1, cj] - dy_safe)

                Y[ri, ci] = y_low + oY[ri, ci] * max(0, ub - y_low)

        # 提取选中单元坐标
        qx = X[sel]; qy = Y[sel]
        return qx, qy

    @staticmethod
    def _expand(qx, qy):
        """第一象限 → 四象限。"""
        Nq = len(qx)
        x = np.empty(4 * Nq); y = np.empty(4 * Nq)
        x[0*Nq:1*Nq] = qx;    y[0*Nq:1*Nq] = qy
        x[1*Nq:2*Nq] = -qx;   y[1*Nq:2*Nq] = qy
        x[2*Nq:3*Nq] = -qx;   y[2*Nq:3*Nq] = -qy
        x[3*Nq:4*Nq] = qx;    y[3*Nq:4*Nq] = -qy
        return x, y


# ═══════════════════════════════════════════════════
#  简单网格选通映射（原 PlanarMapper）
# ═══════════════════════════════════════════════════

class PlanarMapper(BaseMapper):
    """平面阵网格选通映射器 (thinning)。

    在固定矩形网格上，通过 sigmoid 映射决定每个候选阵元的激活状态。
    八邻域硬约束确保无相邻阵元。
    """

    def __init__(self, Nx: int, Ny: int, dx: float, dy: float, dmin: float = 0.5):
        super().__init__(Ne=Nx * Ny, L=max(Nx * dx, Ny * dy), dmin=dmin)
        self._Nx, self._Ny = Nx, Ny
        self._dx, self._dy = dx, dy
        xs = (np.arange(Nx) - (Nx - 1) / 2) * dx
        ys = (np.arange(Ny) - (Ny - 1) / 2) * dy
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)
        self._grid_x = self._grid_x.ravel()
        self._grid_y = self._grid_y.ravel()

    @property
    def n_vars(self) -> int:
        return self._Ne

    def synthesize(self, opt_vector: np.ndarray) -> tuple:
        v = np.asarray(opt_vector, dtype=float)
        prob = 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))
        active = prob > 0.5
        active = self._collision_check(active)
        return self._grid_x[active].copy(), self._grid_y[active].copy()

    def _collision_check(self, active: np.ndarray) -> np.ndarray:
        active_2d = active.reshape(self._Ny, self._Nx)
        for i in range(self._Ny):
            for j in range(self._Nx):
                if not active_2d[i, j]:
                    continue
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self._Ny and 0 <= nj < self._Nx:
                            if active_2d[ni, nj]:
                                if ni < i or (ni == i and nj < j):
                                    active_2d[i, j] = False
                                else:
                                    active_2d[ni, nj] = False
        return active_2d.ravel()
