"""优化运行脚本 — 加载 Config.json → 组装场景 → CMA-ES 优化 → 保存结果 + 绘图。

用法:
  cd examples/linear_sidelobe && python run.py
  cd examples/linear_sidelobe && python run.py --tag experiment1 --seed 42
"""

import sys, time, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np

from antopt import BeamformingConfig
from antopt import assemble_scenario
from antopt import minimize
from antopt import ScenarioResult

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main(config_path=None, tag="", seed=None):
    # ── 1. 配置 ──
    if config_path is None:
        import __main__ as _main
        caller_dir = Path(_main.__file__).resolve().parent if hasattr(_main, '__file__') else Path.cwd()
        config_path = caller_dir / "Config.json"
    config_path = Path(config_path)
    example_name = config_path.parent.name

    cfg = BeamformingConfig.from_json(config_path)

    if seed is None and cfg.randomSeed is None:
        seed = int(np.random.randint(0, 2**31))
    elif seed is None:
        seed = cfg.randomSeed
    cfg.randomSeed = seed

    freqs = np.array(cfg.frequenciesGHz, dtype=float)
    lam0 = 299792458.0 / (freqs[0] * 1e9)

    print(f"=== {example_name} 优化 ===")
    print(f"  频率: {freqs} GHz  |  λ={lam0*1000:.2f}mm  |  seed: {seed}")
    print(f"  theta ∈ [{cfg.theta_start}°, {cfg.theta_end}°], "
          f"theta0 = {cfg.theta0s}")

    # ── 2. 场景 + 优化 ──
    problem, bounds, x0 = assemble_scenario(cfg)
    print(f"  n_vars={problem.n_vars} (pos={problem.n_pos}, "
          f"phase={problem.n_phase}, amp={problem.n_amp})")
    print(f"  激活组件: {list(problem.components.keys())}")

    stop_fit = cfg.opt_stop_fitness
    if stop_fit is None and cfg.target_components:
        stop_fit = 0.0

    print(f"\n--- {cfg.opt_method.upper()} 优化 ---")
    t0 = time.perf_counter()
    result = minimize(
        problem.fitness, problem.n_vars,
        method=cfg.opt_method, bounds=bounds, x0=x0,
        sigma=cfg.opt_sigma, pop_size=cfg.opt_pop_size,
        max_iter=cfg.opt_max_iter, seed=seed,
        n_jobs=cfg.opt_n_jobs, verbose=cfg.opt_verbose,
        stop_fitness=stop_fit,
    )
    elapsed = time.perf_counter() - t0
    print(f"  最优 fitness: {result['f']:.4f}  |  耗时: {elapsed:.1f}s")

    # ── 3. 结果（自动计算指标 + 保存 + 打印 + 绘图）──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sr = ScenarioResult.from_optimization(
        problem, result, cfg=cfg,
        wavelength_m=lam0, elapsed_seconds=elapsed,
        tag=(tag or f"{freqs[0]}GHz_seed{seed}"),
    )
    sr.timestamp = ts
    sr.print_summary()

    decoded = problem.get_result(result["x"])
    saved_dir = sr.save(PROJECT_ROOT / "results" / example_name, wavelength_m=lam0)
    sr.save_figures(problem, decoded, cfg, example_name, saved_dir, wavelength_m=lam0)

    print(f"  结果已保存: {saved_dir}")
    return sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阵列优化")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(config_path=args.config, tag=args.tag, seed=args.seed)

