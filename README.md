# antopt — 阵列天线波束成形优化库

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-66%20passed-green)](#)
[![license](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

> 稀疏/稀布阵列天线幅相优化工具包 — 低副瓣、任意零陷、差波束、方向性约束

## 功能

| 能力 | 说明 |
|---|---|
| **低副瓣** (PSLL) | 全局副瓣抑制，支持主瓣保护区排除 |
| **任意零陷** | 单/多方向零陷，独立保护窗口 |
| **差波束** | 单脉冲差波束，中心零深 + 对称度约束 |
| **方向性约束** | 绝对/相对方向性下限，防增益暴跌 |
| **波束宽度约束** | 半功率波束宽度上限 |
| **线阵 / 面阵** | 线阵 Stick-Breaking 连续位分布，面阵网格选通 |
| **单元方向图** | IEP（共享）和 AEP（逐元）两种合成模式，支持 HFSS 导入 |
| **多频多角度** | 任意频率/扫描角组合，取最差适应度 |
| **空间映射** | ASM / MSM 粗-细模型桥接 |

## 快速开始

```bash
# 安装
pip install -e .

# 运行算例
cd examples/linear_sidelobe
python run.py
```

## 配置示例

```jsonc
// examples/linear_sidelobe/Config.json
{
  "frequenciesGHz": [9],
  "aspectAngle": { "thetaDeg": [-90, 90, 0.1], "theta0sDeg": [0] },
  "position": { "source": "optimize" },
  "amplitude": { "source": "optimize" },
  "phase": { "source": "default" },
  "target": {
    "components": {
      "sidelobe": { "weight": 1.0, "targetDb": -30 },
      "main_lobe_pointing": { "weight": 10 }
    }
  },
  "optimizer": { "method": "cma", "max_iter": 500 }
}
```

## 优化器

| 方法 | 说明 |
|---|---|
| **CMA-ES** | 协方差矩阵自适应进化策略（主力） |
| **DMDE** | 双变异差分进化（前后期 F 值自适应） |
| **GWO** | 灰狼优化器 |

## 六组件代价模型

总适应度 = Σ(w_i × C_i)，各组件代价值恒 ≥ 0：

```
main_lobe_pointing  → (θ_peak − θ₀)²
sidelobe            → max(0, PSLL − PSLL_target)
null_steering       → Σ max(0, P_max_window − P_target)
directivity         → max(0, D_threshold − D)²
hpbw                → max(0, HPBW − HPBW_max)
difference_beam     → null_penalty + symmetry_penalty
```

## 目录结构

```
beamforming2/
├── README.md
├── pyproject.toml
├── antopt/              # 核心库
│   ├── antenna/         #   天线基础（Pattern / Geometry / ElementPattern）
│   ├── mapping/         #   位置映射（LMMapper / PlanarMapper）
│   ├── opt/             #   优化管线（Config → Scenario → Problem → Solver）
│   ├── analysis/        #   方向图分析 + 可视化
│   ├── io/              #   HFSS 读写 + 结果存取
│   └── space_mapping/   #   空间映射（ASM / MSM）
├── tests/               # 单元测试 (66 passed)
├── examples/            # 使用示例
│   ├── linear_sidelobe/
│   ├── linear_composite_null/
│   ├── linear_difference_beam/
│   └── planar_advanced/
├── docs/
│   └── knowledge/       # 天线阵列优化知识库
├── input/               # 输入数据（EP / 阵列配置）
└── scripts/             # 工具脚本
```

## 依赖

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`
- 可选：`cma >= 3`（空间映射 PE）、`scikit-learn >= 1.3`（代理模型）

## 引用

本项目核心计算原理详见 `docs/knowledge/antenna-array-optimization.md`。
