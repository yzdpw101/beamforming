# 研究记录 — beamforming 项目

> 记录关键技术决策、调研过程、基准测试结果和教训。

---

## 2026-06-01 — 项目架构重构 (v0.2)

### 目录重组

| 旧 | 新 | 理由 |
|---|---|---|
| `core/` | `antenna/` | 领域语义更清晰 |
| `matrix_mapping/` | `mapping/` | 去冗余前缀 |
| `analysis.py` (flat) | `analysis/metrics.py` | 归入子包 |
| `plotting.py` (flat) | `analysis/plotting.py` | 与分析同行 |
| `optimizer.py` | `opt/solver.py` | 归入优化管线 |
| `config.py` | `opt/config.py` | 同上 |
| `problem.py` | `opt/problem.py` | 同上 |
| `scenario.py` | `opt/scenario.py` | 同上 |
| `hfss_io.py` | `io/hfss.py` | 归入 IO 层 |
| `result.py` | `io/result.py` | 同上 |

### 输出格式统一

- **`elements.txt`**: 合并 `positions.txt` + `amplitudes.txt` + `phases.txt`
  ```
  arrayType: linear
  elementsNum: 32
               x              y     amplitudes         phases
       -0.240250       0.000000       0.191999       0.000000
  ```
- **`optResult.json`**: 精简，去 `config_snapshot`，增加 `components` 和 `replay`

### 角度精度规则

所有角度值必须四舍五入到 `theta_step` 精度：
```python
val = round(deg / step) * step
val = 0.0 if abs(val) < step * 0.5 else val
```
原因：浮点 -5.1e-12 → 显示为 -0.0°；-66.60000000000001 → 显示不干净。

### matplotlib LaTeX 双反斜杠

Python f-string 中 `\t` 被转义为 TAB。必须用 `\\t`：
```python
f"$\\theta_0$={t0}°"  # ✅
f"$\theta_0$={t0}°"   # ❌ 显示 heta
```

### EP `input_theta_range` 默认值

默认 `(-180, 180)` 导致 CSV 采样步长翻倍，EP 插值全错位，PSLL 偏差 ~2dB。
改为 `(-90, 90)`。此参数在 `element_pattern.py:92`、`config.py:62`、`hfss.py:64` 三处。

### `importInitIndividual` 移入 `optimizer` 段

从 Config.json 顶层移入 `"optimizer": {...}`，对应 `config.py` 字段重命名：
`import_init_individual` → `opt_import_init_individual`

### 画图脚本统一读 `elements.txt`

`load_elements(path)` 作为公共函数放在 `io/result.py`，画图脚本不再各自拷贝解析逻辑。

---

## 2026-06-01 — 平面阵内存爆炸与分块

### 问题

10000 元随机面阵，0.1° 分辨率（Nθ=1801，Nφ=361）：
```python
phase = np.outer(x, du.ravel())  # (10000, 650461) × 8B = 5.2 GB
```
两个 outer + exp 共需 ~26 GB 峰值内存。

### 方案 A：θ 维分块

```
当前： (N, Nθ, Nφ) 全量矩阵
分块后：每块 (N, 50, Nφ)，内存 ~150 MB
```

在 `pattern.py` 中新增 `_planar_af_chunked()` 和 `_planar_af_vec()`。
当 Nθ > 200 时自动启用（小阵列走原向量化路径）。

### 方案 B：元素维分块

对超大规模（N > 5000），在 θ 分块基础上再加元素分块。
同样在 `_planar_af_chunked()` 中实现。

### 方案 C：FFT（仅均匀栅格）

`planar_af_fft()` 适用于均匀矩形栅格，O(Ngrid·log Ngrid) 复杂度。
对 526×126 均匀面阵（66276 元）：1.6 秒，~100 MB。
精度：PSLL 差 0.000dB，RMS 误差 0.07dB（大误差仅出现在 θ > 88° 边缘）。

---

## 2026-06-01 — FINUFFT 调研与验证

### 调研结论

阵列因子 AF(u,v) = Σ A_n · exp(j·2π·(x_n·u + y_n·v)) 本质上就是 **2D Type-3 NUFFT**。

| 库 | 安装 | 加速比 | 精度 | 适用 |
|---|---|---|---|---|
| **FINUFFT** | `pip install finufft` | 20-2700× | 0.000dB | 任意非均匀 |
| FFT (planar_af_fft) | 内置 | 快 | 0.07dB RMS | 仅均匀栅格 |

### 关键公式（Type 3 缩放）

```python
L = max(max(|x|), max(|y|))           # 公共归一化系数
xs, ys = x * (π/L), y * (π/L)         # 源坐标 → [-π, π]
ss = 2*L * sinθ·cosφ                  # 目标坐标
ts = 2*L * sinθ·sinφ
af = finufft.nufft2d3(xs, ys, c, ss, ts, eps=1e-6)
```

推导：exp(i·x_s·s) = exp(i·(x·π/L)·(2L·u)) = exp(i·2π·x·u) ✓

### 基准测试结果

**单次计算**（1° θ步长，2° φ步长）：

| 配置 | 阵元 | 直接 | NUFFT | 加速 | 误差 |
|---|---|---|---|---|---|
| 线阵 均匀 | 500 | 0.25s | 0.013s | 20× | 0.000dB |
| 线阵 稀布 | 500 | 0.26s | 0.009s | 28× | 0.000dB |
| 面阵 均匀 32×32 | 1024 | 0.54s | 0.006s | 94× | 0.001dB |
| 面阵 稀布 | 1000 | 0.55s | 0.011s | 49× | 0.000dB |
| 面阵 稀布 | 5000 | 2.74s | 0.010s | 276× | 0.000dB |
| 面阵 稀布 | 10000 | 5.54s | 0.008s | 690× | 0.000dB |

**多种群模拟**（CMA-ES 场景，固定位置变激励）：

| 配置 | 阵元 | pop | 直接 | NUFFT | 加速 |
|---|---|---|---|---|---|
| 线阵 稀布 | 500 | 100 | 29.2s | 0.7s | 39× |
| 面阵 稀布 | 1000 | 100 | 63.0s | 1.0s | 63× |
| 面阵 稀布 | 5000 | 50 | 136.8s | 0.3s | 448× |

**结论**：FINUFFT Type 3 在任意阵型（线/面、均匀/非均匀）下均可替代直接求和，精度无损（误差<0.001dB），加速 20-2700×。

### 集成到 pattern.py

**已完成 (2026-06-01)**：

- [x] `pattern.af(method="auto")` — 自动选 NUFFT/直接
- [x] `pattern.af_nufft()` — 强制 NUFFT，多频多角度自适应
- [x] `_nufft_1d()` / `_nufft_2d()` — 底层 Type 3 封装
- [x] `BeamformingProblem.af_method` — Config.json 控制
- [x] `Config.json` 新增 `"optimizer": { "afMethod": "auto" }`
- [x] 测试：66/66 通过，NUFFT vs 直接误差 < 0.0001dB

**线阵 1D 公平对比（补充）：**

| 元数 | linear_af | NUFFT 1D | 加速 |
|---|---|---|---|
| 500 | 0.003s | 0.009s | — |
| 5000 | 0.026s | 0.0035s | 7.4× |

小阵列（<300元）自动回落直接计算，无 NUFFT 开销。

### 待验证项

- [ ] FINUFFT Plan 接口（Type 3 的 many 模式）
- [ ] 极低精度（eps=1e-3）下是否可用于早期优化迭代

---

## 2026-06-01 — 旧项目清理

### 删除的垃圾文件

`build/`, `dist/`, `KTY_面阵优化.spec`, `_test_pos.png`, `results/_test_*`, `.codewhale/`, `hfss/`, `releases/`

### 新增基础设施

`.gitignore`, `README.md`, `scripts/`

---

## 调试教训

1. **`\t` 转义**：Python f-string 中所有 LaTeX `\t` 需写成 `\\t`
2. **`input_theta_range`**：默认值 -180~180 和实际 CSV 数据范围 -90~90 不匹配
3. **`amp_in_db`**：elements.txt 存线性值，画图脚本默认值应 `False`
4. **`result_dir` 路径**：用户可能传 `elements.txt` 文件路径而非目录，需 `is_file()` 检测
5. **GitHub push**：Windows + 代理环境需 `git -c http.proxy=... -c https.proxy=... push`
