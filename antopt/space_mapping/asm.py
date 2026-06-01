"""渐进空间映射 (ASM) — 粗模型(方向图乘积) + 细模型(HFSS) 迭代对齐。"""

import numpy as np
from .base import calc_response_error, compute_fine_psll


def parameter_extraction(Xf, Yf_list, mapper, pattern, fe_patterns,
                         pop_size=50, max_iter=200, sigma=0.3, seed=0, verbose=False):
    """CMA-ES 找 Xe 使粗模型响应逼近细模型 (对应 C++ PE)。

    注：需要安装 cma 包 (`pip install cma>=3`)。
    """
    try:
        import cma
    except ImportError:
        raise ImportError("参数提取需要 cma 包: pip install cma>=3")

    n_vars = mapper.n_vars

    def objective(x):
        pos = mapper.synthesize(x)
        from ..utils import compute_pattern
        Yc = compute_pattern(pos, np.ones(len(pos)), np.zeros(len(pos)),
                             mapper, pattern, fe_patterns)
        return calc_response_error([Yc], Yf_list)

    x0 = Xf.copy() if Xf is not None else np.ones(n_vars) * 0.5
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': pop_size, 'maxfevals': pop_size * max_iter,
        'seed': seed, 'verbose': -9 if not verbose else 1,
    })
    es.optimize(objective)
    return es.result.xbest


def run_space_mapping(cfg, mapper, pattern, fe_patterns,
                       hfss_eval_func, coarse_model_func,
                       verbose=True):
    """渐进空间映射主循环 (预留)。

    Args:
        cfg: ASMConfig 实例
        mapper: LMMapper
        pattern: Pattern
        fe_patterns: 单元方向图
        hfss_eval_func: 细模型评估函数
        coarse_model_func: 粗模型评估函数
        verbose: 是否打印日志

    Returns:
        (X_opt, result_dict)
    """
    raise NotImplementedError("ASM 主循环待实现")
