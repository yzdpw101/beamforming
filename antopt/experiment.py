"""实验运行器 — 单次 / 批量 / 网格搜索。"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .opt.config import BeamformingConfig
from .opt.scenario import assemble_scenario
from .opt.solver import minimize
from .io.result import ScenarioResult


def run_single(cfg: BeamformingConfig, seed: Optional[int] = None,
               tag: str = "", output_dir: Optional[str] = None) -> "ScenarioResult":
    """单次优化运行。

    Args:
        cfg: 配置对象
        seed: 随机种子 (覆盖 cfg.randomSeed)
        tag: 结果标签
        output_dir: 输出目录, None=不保存

    Returns:
        ScenarioResult
    """
    if seed is not None:
        cfg.randomSeed = seed

    problem, bounds, x0 = assemble_scenario(cfg)
    lb, ub = bounds

    t0 = time.perf_counter()
    result = minimize(
        problem.fitness,
        n_vars=problem.n_vars,
        method=cfg.opt_method,
        bounds=(lb, ub),
        sigma=cfg.opt_sigma,
        pop_size=cfg.opt_pop_size,
        max_iter=cfg.opt_max_iter,
        n_jobs=cfg.opt_n_jobs,
        verbose=cfg.opt_verbose,
        seed=cfg.randomSeed,
        stop_fitness=cfg.opt_stop_fitness,
        x0=x0,
    )
    elapsed = time.perf_counter() - t0

    wavelength_m = 299792458.0 / (cfg.frequenciesGHz[0] * 1e9)
    sr = ScenarioResult.from_optimization(
        problem, result, cfg=cfg,
        wavelength_m=wavelength_m,
        elapsed_seconds=elapsed,
        tag=tag,
    )

    if output_dir:
        sr.save(Path(output_dir), wavelength_m)

    return sr


def run_batch(cfg: BeamformingConfig, seeds: list[int],
              tags: Optional[list[str]] = None,
              output_dir: Optional[str] = None) -> list["ScenarioResult"]:
    """多 seed 批量运行。

    Args:
        cfg: 基础配置
        seeds: 随机种子列表
        tags: 对应标签, None=用 seed 作标签
        output_dir: 输出根目录

    Returns:
        list of ScenarioResult
    """
    results = []
    for i, seed in enumerate(seeds):
        tag = tags[i] if tags else f"seed{seed}"
        print(f"\n=== seed={seed} ({i+1}/{len(seeds)}) ===")
        sr = run_single(cfg, seed=seed, tag=tag, output_dir=output_dir)
        results.append(sr)
        print(f"  PSLL={sr.psll:.2f} dB, fitness={sr.fitness:.4f}" if sr.psll else
              f"  fitness={sr.fitness:.4f}")
    return results


def collect_results(results_dir: Path) -> list[dict]:
    """扫描结果目录，提取关键指标。

    Args:
        results_dir: 包含子目录（每次运行一个）的结果根目录

    Returns:
        [{"timestamp": ..., "psll_db": ..., "directivity_dbi": ..., "fitness": ...}, ...]
    """
    import json
    collected = []
    for subdir in sorted(Path(results_dir).glob("*")):
        json_path = subdir / "optResult.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_dir"] = str(subdir)
            collected.append(data)
    return collected
