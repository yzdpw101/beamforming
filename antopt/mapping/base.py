"""位置映射抽象基类。"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseMapper(ABC):
    """将优化变量映射为物理阵元位置的抽象接口。

    线阵使用 Stick-Breaking（LMMapper），平面阵使用网格选通（PlanarMapper）。
    """

    def __init__(
        self,
        Ne: int,
        L: float,
        dmin: float,
        dmax: Optional[float] = None,
        is_symmetric: bool = False,
        is_fixed_aperture: bool = False,
    ):
        self._Ne = Ne
        self._L = L
        self._dmin = dmin
        self._dmax = dmax
        self._is_symmetric = is_symmetric
        self._is_fixed_aperture = is_fixed_aperture

    @property
    def Ne(self) -> int:
        """阵元总数。"""
        return self._Ne

    @property
    def L(self) -> float:
        """孔径总长度（波长单位）。"""
        return self._L

    @property
    def dmin(self) -> float:
        """最小阵元间距（波长单位）。"""
        return self._dmin

    @property
    def dmax(self) -> float:
        """最大阵元间距（波长单位），None = 无上限。"""
        return self._dmax if self._dmax is not None else np.inf

    @property
    def is_symmetric(self) -> bool:
        """是否关于原点对称。"""
        return self._is_symmetric

    @property
    def is_fixed_aperture(self) -> bool:
        """是否固定孔径。"""
        return self._is_fixed_aperture

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """优化变量维度。"""
        ...

    @abstractmethod
    def synthesize(self, opt_vector: np.ndarray) -> np.ndarray:
        """将优化变量映射为阵元位置。

        Returns:
            阵元位置数组, shape (Ne,), 已排序, 波长单位
        """
        ...
