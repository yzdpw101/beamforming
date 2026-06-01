# 阵列天线波束成形优化 — 核心知识库 (组件化多约束版)

本文件是 `beamforming` 项目的核心数学与物理原理说明书。库内部核心计算层全面采用**组件化标量化复合代价模型**，不再对任务类型进行硬编码分类。

---

## 1. 远场辐射与阵列天线数学基础

### 1.1 二维有源相量叠加公式
任何矩形平面阵列在远场特定方向 $(\theta, \phi)$ 产生的总辐射场强 $E(\theta, \phi)$ 是各阵元有源辐射场的相量叠加结果：

$$E(\theta,\phi) = \sum_{n=0}^{N-1} A_n \cdot Fe_n(\theta,\phi) \cdot e^{j \left[ k_0 (x_n \sin \theta \cos \phi + y_n \sin \theta \sin \phi) + \psi_n \right]}$$

- $N$: 阵列的总阵元数。若为常规线阵，则退化为 $N_y = 1, y_n = 0$。
- $A_n$: 第 $n$ 个阵元的激励线性幅度。
- $\psi_n$: 第 $n$ 个阵元的激励相位（弧度，$\text{rad}$）。
- $x_n, y_n$: 第 $n$ 个阵元在 $xy$ 平面上的物理位置，严格以工作波长 $\lambda_0$ 为单位（即 $pos\_wl$）。
- $k_0 = 2\pi$: 自由空间归一化波数（由于位置已归一化为波长，此处 $k_0$ 恒等于 $2\pi$）。
- $Fe_n(\theta, \phi)$: 第 $n$ 个阵元在该方向的有源单元方向图（AEP，Active Element Pattern），包含了互耦效应。

### 1.2 单元方向图（EP）集成模式
系统通过矩阵广播机制支持以下两种单元方向图合成：
- **共享 EP 模式 (IEP)**：全阵共用一个隔离单元方向图，场强公式简化为方向图乘积定理：
  $$E(\theta,\phi) = Fe(\theta,\phi) \times AF(\theta,\phi)$$
  其中阵因子为：
  $$AF(\theta,\phi) = \sum_{n=0}^{N-1} A_n \cdot e^{j \left[ 2\pi (x_n \sin \theta \cos \phi + y_n \sin \theta \sin \phi) + \psi_n \right]}$$
- **逐元 EP 模式 (AEP)**：每个阵元拥有独立从 HFSS 提取的 CSV 方向图数据，必须严格通过相量级进行逐元叠加：
  $$E(\theta,\phi) = \left| \sum_{n=0}^{N-1} \left[ Fe_n(\theta,\phi) \cdot A_n \cdot \cos(\Phi_n) + j \cdot Fe_n(\theta,\phi) \cdot A_n \cdot \sin(\Phi_n) \right] \right|$$
  其中 $\Phi_n = 2\pi (x_n \sin \theta \cos \phi + y_n \sin \theta \sin \phi) + \psi_n$。

### 1.3 幅度分贝极性与转换
在配置层（`Config.json`）中，幅度通常以 dB 形式约束，且遵循**“正 dB 表示衰减”**的工程约定。
若设置边界 `bounds = [0, 20]`，代表最大衰减 **20 dB**。其向线性幅度的转换公式为：

$$A_n = 10^{-\frac{dB_n}{20}}$$

即 $0\text{ dB} \rightarrow 1.0$（不衰减），$20\text{ dB} \rightarrow 0.1$（衰减到十分之一场强）。

---

## 2. 全物理坐标系体系与 IO 隔离机制

### 2.1 全物理约定定义

天线远场方向的完整空间定义为 $\theta \in [0^\circ, 180^\circ]$、$\phi \in [0^\circ, 360^\circ]$，库内部算法层、分析层及组件层统一采用此标准球面物理坐标系：

| 物理维度 | 取值范围 | 空间物理意义 |
|:---|:---|:---|
| $\theta$ | $[0^\circ, 180^\circ]$ | 天线法线指向（$z$ 轴正方向）为 $\theta = 0^\circ$；天线所在平面（$xy$ 平面）为 $\theta = 90^\circ$。 |
| $\phi$ | $[0^\circ, 360^\circ]$ | 自 $x$ 轴正方向向 $y$ 轴正方向逆时针旋转的角度。 |
| 上半球空间 | $\theta \in [0^\circ, 90^\circ]$ | 天线的主要辐射前向半空间。 |

**E 面/H 面的简便描述约定**：天线工程中经常只需要观察特定切面（如 E 面 $\phi = 0^\circ$）。若严格按完整定义，描述 E 面需要 $\phi = 0^\circ$ 和 $\phi = 180^\circ$ 两个切面的 $\theta \in [0^\circ, 180^\circ]$ 拼接。为简便，工程上引入 $\phi = 0^\circ, \theta \in [-90^\circ, 90^\circ]$ 的记法，其等价关系为：

$$\phi = 0^\circ, \theta \in [-90^\circ, 90^\circ] \quad \Longleftrightarrow \quad \phi \in \{0^\circ, 180^\circ\}, \theta \in [0^\circ, 180^\circ]$$

转换规则：当 $\theta < 0^\circ$ 时，取 $|\theta|$ 并将 $\phi$ 加 $180^\circ$（或 $\phi$ 不变、$\theta$ 取绝对值后补 $\phi+180^\circ$ 切面）。

**输入参数的简便处理**：用户在 `Config.json` 中输入角度范围时，同样允许使用简便记法（如 $\theta \in [-30^\circ, 90^\circ]$）。库内部仅在检测到范围重叠（如同时指定了 $\phi = 0^\circ$ 和 $\phi = 180^\circ$ 且 $\theta$ 范围覆盖了正负区间）时发出警告，随后自动转换为标准物理网格。

**非对称 $\theta$ 范围的注意**：若原始 $\theta$ 关于原点不对称（例如 $\theta \in [-30^\circ, 90^\circ]$、$\phi \in [0^\circ, 180^\circ]$），转换后变为两个不连续块：
- $\theta \in [0^\circ, 90^\circ], \phi \in [0^\circ, 180^\circ]$
- $\theta \in [0^\circ, 30^\circ], \phi \in [180^\circ, 360^\circ]$

拼接后的角度网格**不是完整矩形**——部分 $(\theta, \phi)$ 点为空（无数据）。此时若进行方向性系数积分，还需考虑块边界重复点（$\phi = 0^\circ / 180^\circ / 360^\circ$ 处的重叠）的权重降低。

> **适用场景**：对于线阵（仅 $\phi = 0^\circ$ 面）或矩形平面阵的 $\phi = 0^\circ$ / $\phi = 90^\circ$ 切面，计算和绘图直接使用简便记法即可，无特殊处理。若优化目标涉及**非对称 $\theta$ 的全空间优化**，则计算与绘图前必须仔细处理角度拼接与边界权重问题。

### 2.2 IO 层的解耦职责
从 Ansys HFSS 导出的原始 CSV 文件往往采用非标约定（例如 $\theta \in [-90^\circ, 90^\circ], \phi = 0^\circ$）。
- **硬性规则**：此数据在被 `hfss_io.py` 加载的瞬间，必须立即利用对称性或重组逻辑转换为 $\theta \in [0^\circ, 180^\circ]$ 且 $\phi \in [0^\circ, 360^\circ]$ 的标准二维物理网格矩阵。
- **收益**：核心优化计算模块（`antopt/`）接收到的全部是纯净的物理矩阵，内部严禁出现任何 `if hfss_mode:` 的运行时判断分支。

---

## 3. 组件化复合代价模型（加权标量化）

优化器核心（如 CMA-ES）通过最小化一个标量代价值（Fitness）来驱动进化。总适应度函数 $F(x)$ 是所有被激活组件输出代价值的**加权求和**：

$$F(x) = \sum_{i \in \text{Activated}} w_i \cdot C_i(ctx)$$

为了使优化终止条件 `stopFitness = 0.0` 具备清晰的物理判定意义（即所有硬约束完美达标且指标最优时停止），**所有组件的代价值 $C_i$ 必须设计为恒 $\ge 0$ 的形式**。

评估上下文 `EvaluationContext`（简称 `ctx`）会预先计算出当前个体的归一化对数功率方向图 $P_{dB}(\theta, \phi)$，并定位实际主瓣峰值位置 $(\theta_{peak}, \phi_{peak})$。

### 3.1 主瓣指向对准组件 (`main_lobe_pointing`)
- **目的**：确保实际波束的最大辐射方向精准对准设计目标指向 $(\theta_0, \phi_0)$。
- **数学模型**：计算实际峰值与目标指向之间的角度方差：
  $$C_{\text{pointing}} = (\theta_{peak} - \theta_0)^2 + (\phi_{peak} - \phi_0)^2$$

### 3.2 副瓣优化组件 (`sidelobe`)
- **目的**：抑制主瓣区以外的全局旁瓣电平。
- **数学模型**：若配置了主瓣保护区半宽 $\theta_{null\_width}$（非零值），则在球面网格中剔除该主瓣保护区 $[\theta_0 - \theta_{null\_width}, \theta_0 + \theta_{null\_width}]$，得到旁瓣区网格 $\Omega_{S}$。**当 $\theta_{null\_width}$ 为 `null` 或 `0` 时，表示不设置主瓣保护区**，整个可见空间全部参与旁瓣搜索。提取旁瓣区内的全局最高峰值 $PSLL_{current}$：
  $$PSLL_{current} = \max_{(\theta, \phi) \in \Omega_{S}} P_{dB}(\theta, \phi)$$
  $$C_{\text{sidelobe}} = \max \left( 0, PSLL_{current} - PSLL_{target} \right)$$
  若要追求极致副瓣，可将 $PSLL_{target}$ 设为 $-100\text{ dB}$，使其退化为持续推进模式。

### 3.3 零陷约束组件 (`null_steering`)
- **目的**：在指定的干扰方向上压低辐射，支持任意数量、任意角度的单/多零陷。
- **数学模型**：给定一组零陷中心角 $\left\{(\theta_m, \phi_m)\right\}$ 和保护半窗宽 $\Delta$（度），在窗口内提取最大增益：
  $$\Omega_{null, m} = \left\{ (\theta, \phi) \mid \theta \in [\theta_m - \Delta, \theta_m + \Delta], \phi \in [\phi_m - \Delta, \phi_m + \Delta] \right\}$$
  $$P_{max, m} = \max_{(\theta, \phi) \in \Omega_{null, m}} P_{dB}(\theta, \phi)$$
  $$C_{\text{null}} = \sum_{m} \max \left( 0, P_{max, m} - P_{target, m} \right)$$

### 3.4 方向性系数约束组件 (`directivity`)
- **目的**：防止因追求极致旁瓣或零陷而导致阵列绝对增益暴跌。
- **数学模型**：通过懒加载获取当前个体的绝对方向性系数 $D_{current}$（计算见第 4 节）。
  - 绝对模式 ("abs")：直接给定硬门槛 $D_{threshold} = D_{target}$。
  - 相对模式 ("rel")：基于全 1 激励的阵列参考方向性系数 $D_{ref}$ 允许其下降 $\Delta D$：$D_{threshold} = D_{ref} - \Delta D$。
  $$C_{\text{directivity}} = \max \left( 0, D_{threshold} - D_{current} \right)^2$$
  使用平方惩罚以在违背约束时施加高额代价值。

### 3.5 主瓣宽度约束组件 (`hpbw`)
- **目的**：约束半功率波束宽度（HPBW），防止主瓣展宽严重。
- **数学模型**：沿特定的切面（如 $\phi = 0^\circ$ 或 $\phi = 90^\circ$），自实际峰值 $\theta_{peak}$ 向两侧搜索功率下降到 $-3\text{ dB}$ 的两个拐点 $\theta_{left}$ 和 $\theta_{right}$。
  $$HPBW_{current} = |\theta_{right} - \theta_{left}|$$
  $$C_{\text{hpbw}} = \max \left( 0, HPBW_{current} - HPBW_{max} \right)$$

### 3.6 差波束组件 (`difference_beam`)
- **目的**：形成在目标指向中心处具有尖锐零陷、两侧对称抬起的单脉冲差波束。
- **数学模型**：将阵列划分为左/右或上/下两部分。计算其差方向图（通过引入 $180^\circ$ 倒相）。其代价函数包含中心点零陷深度和两侧尖峰对称度：
  $$C_{\text{diff}} = \max\left(0, P_{dB}(\theta_0, \phi_0) - P_{diff\_null\_target}\right) + \left| P_{dB}(\theta_{peak\_left}) - P_{dB}(\theta_{peak\_right}) \right|$$

---

## 4. 方向性系数数值积分核心算法与边界补偿

天线绝对方向性系数 $D$ 的物理定义为最大辐射强度与空间平均辐射功率之比：

$$D = \frac{4\pi \cdot \max |E(\theta,\phi)|^2}{P_{rad}}$$

其中总辐射功率 $P_{rad}$ 是通过在闭合球面（或前向半球面）上对功率方向图进行二维数值积分得到的：

$$P_{rad} = \int_{0}^{2\pi} \int_{0}^{\theta_{max}} |E(\theta,\phi)|^2 \sin \theta \,d\theta \,d\phi$$

最终结果转换为 dBi：$D_{dBi} = 10 \log_{10}(D)$。

> **参考实现**：`diver.m` 提供了标准的 MATLAB 梯形积分（`trapz`）实现，采用先 $\theta$ 后 $\phi$ 的嵌套积分顺序：
>
> ```matlab
> TotalRadiation = trapz(phi, trapz(theta, RadiationPattern .* sin(THETA), 2));
> Directivity = 4 * pi * max(RadiationPattern(:)) / TotalRadiation;
> Directivity_dBi = 10 * log10(Directivity);
> ```
>
> 注意该参考实现中辐射方向图已被归一化（`max = 1`），因此分子简化为 $4\pi$。本项目的 Python 实现基于同一数学原理，但采用矩阵权重逐元点乘求和（见 4.2 节）以避免嵌套循环。

### 4.1 数值积分三大铁律
在编写 `compute_directivity` 函数时，必须无条件通过离散矩阵严格落实以下三条规则，任何一处违反都会产生确定性的系统浮点误差。

1. **输入极性规范**：
   积分式中的 $|E(\theta, \phi)|^2$ 必须是**未归一化的线性场强绝对值的平方**，绝对禁止直接传入对数分贝值（dB）参与积分。

2. **$\theta$ 维度的离散端点半权处理**：
   采用标准梯形法则进行数值离散时，若 $\theta$ 的离散采样点为 $\theta_0, \theta_1, \dots, \theta_{M}$，其积分步长为 $\Delta\theta$。积分权重向量 $W_\theta$ 的首尾两端因属于网格边界，其几何面积权重必须乘以 **0.5**。

3. **$\phi = 0^\circ / 360^\circ$ 物理重叠补偿（极其重要）**：
   在标准三维空间中，$\phi = 0^\circ$ 射线面与 $\phi = 360^\circ$ 射线面在空间几何上是完全重叠的同一个切面。
   如果采样网格中同时包含了这两个端点（例如 `phi = np.arange(0, 360.1, 1)`），直接进行简单的矩阵求和会导致这一方位角上的辐射功率被**重复累加了两次**，这会人为使算出的总功率 $P_{rad}$ 偏大，从而导致最终的天线方向性系数 $D$ 偏小。
   **解决算法**：在构建 $\phi$ 维度的离散积分权重向量 $W_\phi$ 时，必须执行以下强制修正：
   ```python
   W_phi = np.ones(len(phi_array)) * delta_phi
   if np.isclose(phi_array[0], 0.0) and np.isclose(phi_array[-1], 360.0):
       W_phi[0] *= 0.5
       W_phi[-1] *= 0.5
   ```

### 4.2 离散矩阵求和公式

最终在代码中，通过将二维线性功率矩阵 $P_{mat} = |E|^2$ 与由 $\sin\theta$ 及离散权重构成的二维权重矩阵 $W_{mat}$ 进行逐元素点乘（Element-wise product）并全量求和，实现极高效率的向量化积分：

$$W_{mat}[m, n] = W_\theta[m] \cdot W_\phi[n] \cdot \sin(\theta_m)$$

$$P_{rad} = \sum_{m} \sum_{n} P_{mat}[m, n] \cdot W_{mat}[m, n]$$

---

## 5. 优化变量编码、映射与量化机制

### 5.1 变量向量拓扑结构

为了保证同一种群在优化器中可以无缝交叉与变异，全项目所有个体的编码拓扑结构严格遵循以下拼接顺序，外部禁止手动改变：

$$x = [\underbrace{v_{pos, 0}, \dots, v_{pos, K-1}}_{\text{位置映射变量}(0 \sim 1)} \mid \underbrace{\psi_0, \dots, \psi_{M-1}}_{\text{相位优化变量}(0 \sim 2\pi)} \mid \underbrace{a_0, \dots, a_{L-1}}_{\text{幅度优化变量}}]$$

### 5.2 线阵非均匀位置映射（Stick-Breaking）

> **适用范围**：本节所述 Stick-Breaking 位置映射方法**仅适用于直线阵（Ny=1）的非均匀（稀布）位置优化**。平面阵的位置映射采用不同的机制（见 5.3 节及 `matrix_mapping/planar.py`）。

在进行直线阵稀布（Sparse/Aperiodic）位置优化时，必须严格遵守两步走演进逻辑：

1. **前置容器识别**：在进行连续空间的字典序映射第二步之前，必须优先识别出“确定为空（Destined Empty）”或“确定为满（Destined Full）”的边界容器状态，直接锁定这些网格，不参与后续的自由度松弛。
2. **Stick-Breaking 映射数学步序**：对于剩余的可优化非均匀间距变量 $v_i \in [0, 1)$，其转换为物理波长位置的递推公式为：

$$x_{0} = x_{min}$$

$$x_{i+1} = x_i + d_{min} + R_i \cdot v_i, \quad R_{i+1} = R_i \cdot (1 - v_i)$$

其中总剩余可自由分配的孔径初始值为 $R_0 = L_{total} - (N-1) \cdot d_{min}$。

### 5.3 平面阵网格生成器的八邻域硬约束

在进行二维平面阵列的网格代生成时，为了防止高密度优化下两个相邻阵元靠得太近引发极其严重的强互耦或者物理干涉，网格生成器必须严格执行以下规则：

- **硬性约束**：必须确保在任何一个已放置阵元的**八邻域网格细胞（一圈相邻单元）内，绝对不允许存在另一个阵元**。
- **解耦防呆**：此逻辑必须通过局域碰撞检查动态维持，**严禁简化为固定的 Checkerboard（棋盘格）分块模式**，以确保稀布阵列的随机自由度最大化。

### 5.4 幅相独立离散量化公式

优化器在连续的浮点空间中进化，但为了模拟真实的数字移相器（如 5-bit 移相器，$5.625^\circ$ 步进）和数字衰减器，在构建 `EvaluationContext` 场强计算前，必须对个体变量执行下述阶梯量化：

$$var_{quantized} = \text{round}\left(\frac{var_{continuous}}{step}\right) \times step$$

**相位变量边界硬约束**：相位变量的原始边界为 $[0^\circ, 360^\circ]$，优化器内部对越界变量执行 clamp 修复（直接截断到边界）。问题在于：当个体变量趋近 $360^\circ$ 附近时，一旦因变异/交叉稍微越界就被 clamp 到精确的 $360^\circ$；同理，趋近 $0^\circ$ 的变量越界也被 clamp 到精确的 $0^\circ$。虽然 $0^\circ$ 与 $360^\circ$ 物理上完全等价，但 clamp 行为导致同一个体中可能同时出现两个端点值，造成搜索空间的退化冗余。因此**硬性约束为 $[0, 360^\circ - step]$**——将上界缩小一个步进，使 clamp 目标落在 $360^\circ - step$ 而非 $360^\circ$，从根本上切断重复。

> **备选方案：周期边界修复**：另一条路是不缩小边界，而是保持 $[0^\circ, 360^\circ]$，将越界 clamp 替换为周期 wrap-around——例如更新后变量值变为 $360^\circ + x$，则修复为 $x$（需满足步进量化要求）。此方案保持了范围的物理完整性（$360^\circ$ 自动绕回 $0^\circ$），但需要在优化器边界处理逻辑中额外实现周期映射，复杂度稍高。当前项目采用缩小上界的方案更简单直接。

所有的组件代价评估均基于 $var_{quantized}$ 进行，确保优化结果能直接对接实际工程馈网。

---

## 6. 工程陷阱与调试指南

### 6.1 网格长度差 1 报错 (Shape Mismatch)

- **现象**：在执行 `AF * EP` 时，频繁触发 Python 的维度不匹配错误（如 `(181, 361) vs (180, 361)`）。
- **根源**：在计算阵因子和从 CSV 加载单元方向图时，各自在底层独立调用了 `np.arange(0, 180 + step, step)`。由于浮点数微小的精度漂移，导致生成的数组长度偶尔差 1。
- **消灭手段**：全项目确立**单源网格传递机制**。由 `scenario.py` 统筹生成唯一的 $\theta$ 和 $\phi$ 网格对象，作为入参强制向下分发给 `Pattern` 类和 `hfss_io` 模块，严禁底层私自衍生网格。

### 6.2 Windows 多进程死锁 (Spawn Deadlock)

- **现象**：开启多进程加速后，程序卡死无响应，或者内存瞬间爆炸，无限递归启动主程序。
- **根源**：Windows 操作系统下，Python 进程的分支机制采用 `spawn`。如果在底层计算模块（如 `antopt/core/pattern.py`）内部直接调用 `multiprocessing.ProcessPoolExecutor`，会导致每个子进程重新加载上下文中触发无限死锁。
- **消灭手段**：核心库 `antopt` 内部的循环或加速一律限制使用基于线程的 `ThreadPoolExecutor`（由于主要计算由底层封装的 NumPy 矩阵 C 级并行完成，释放了 GIL 锁，线程池同样高效）。而跨算例或跨频率级别的多进程，必须且只能放置在最外层的 `examples/run.py` 的 `if __name__ == '__main__':` 块内统筹创建。

### 6.3 物理尺度单位爆炸

- **现象**：优化出的阵元位置严重超出合理孔径，方向图呈现极其密集的栅瓣。
- **根源**：没有分清米（m）**与**波长（$\lambda_0$）的界限。`Config.json` 为了方便用户输入，存储的是米；而算法底层的指数项 $j 2\pi x \sin\theta$ 必须要求 $x$ 是波长。
- **消灭手段**：在 `scenario.py` 读取配置后，立刻除以 $\lambda_0$ 转换为波长尺度；在 `result.py` 最终保存到 `positions.txt` 落盘前，立刻乘以 $\lambda_0$ 转换回米，确保核心计算库内部的单位纯净。

---

## 7. 性能优化原则：分支优先于简洁

在核心计算热路径（阵因子、方向图合成、方向性系数积分等）中，**不允许为了代码简洁而牺牲性能**。以下场景必须使用显式 `if` 分支选择快速计算路径：

### 7.1 默认幅相跳过

若当前个体的幅度全为 $1$（线性）且相位全为 $0$，则 $A_n \cdot e^{j\psi_n} \equiv 1$，相乘是冗余操作。必须在进入累加循环前检测此条件，直接跳过幅相乘法：

```python
if np.allclose(amplitudes, 1.0) and np.allclose(phases, 0.0):
    # 快速路径：跳过幅度和相位乘法
    AF = compute_af_positions_only(positions, theta, phi)
else:
    # 标准路径：完整乘加
    AF = compute_af_full(positions, amplitudes, phases, theta, phi)
```

### 7.2 对称阵列快速计算

若阵列满足几何对称条件（关于原点对称且激励对称），阵因子可简化为余弦和形式，计算量减半：

$$AF(\theta,\phi) = \sum_{i=0}^{N/2-1} 2 A_i \cdot \cos(k x_i \sin\theta \cos\phi + \psi_i)$$

检测到对称条件满足时必须走此快速路径。

### 7.3 单元方向图未加载时跳过

若未加载 AEP/IEP 数据（即纯阵因子模式），方向图计算和方向性系数积分均不应执行与 EP 相乘的冗余分支。方向性系数计算中，若 EP 为 None，直接基于 $|AF|^2$ 积分，不引入单元方向图相关的额外数组广播。

### 7.4 总体原则

- 在 `Scenario` 组装阶段做一次特征检测（是否对称、是否默认幅相、是否加载 EP），将结果作为 flag 传入计算核心
- 计算核心内部依据 flag 选择分支，避免运行时在热循环内重复判断
- 允许为不同分支维护独立函数（如 `_compute_af_symmetric()` vs `_compute_af_general()`），不追求"一个函数覆盖所有情况"

---

## 8. 波束显示与结果可视化工具

### 8.1 线阵波束显示（`plot_linear_beam.py`）

**输入**：

- 上次优化结果目录（包含 `amplitudes.txt`、`phases.txt`、`positions.txt`）
- $\theta$ 范围、目标指向 $\theta_0$
- AEP/IEP 加载目录（`null` 表示不加载单元方向图，纯阵因子模式）
- 可选：是否显示 3D 方向图

**可标记/显示的参数**：最大副瓣电平 (PSLL)、主瓣指向、零陷位置与深度、主瓣宽度 (HPBW)、差波束相关指标。

**出图规则**（使用 `subplots` 子图布局）：

1. 第一张：原始方向图（无任何标记）
2. 后续每张：单独标记某一类参数（如仅标 PSLL、仅标零陷、仅标 HPBW 等）
3. 最后一张：所有标记叠加在同一张图上

**打印输出**：所有标记参数的具体数值 + 方向性系数 $D_{dBi}$（方向性系数仅打印，不绘图）。

### 8.2 平面阵波束显示（`plot_planar_beam.py`）

**输入**：

- 上次优化结果目录（包含 `amplitudes.txt`、`phases.txt`、`positions.txt`）
- $\theta$ 范围、$\phi$ 范围、目标指向 $(\theta_0, \phi_0)$
- AEP/IEP 加载目录（`null` 表示不加载单元方向图，纯阵因子模式）
- 可选：指定要显示的 $\phi$ 切面列表（必须在 $\phi$ 采样点中，若不在则警告并向下取整到最近采样点）

**默认显示**：3D 方向图。

**可选 2D 切面图**：若指定了 $\phi$ 切面，则对每个切面生成 2D 方向图（$\theta$ 为横轴）。可标记的参数与线阵相同：PSLL、主瓣指向、零陷、HPBW、差波束。

> **注意**：这些 2D 图仅为单个 $\phi$ 切面，不代表三维全空间的旁瓣/零陷最差情况。

**出图规则**（每个 $\phi$ 切面独立使用 `subplots` 子图布局）：

1. 第一张：该切面的原始方向图（无标记）
2. 后续每张：单独标记某一类参数
3. 最后一张：所有标记叠加

**打印输出**：各切面的标记参数数值 + 全空间方向性系数 $D_{dBi}$（仅打印，不绘图）。

---

## 9. 项目架构与设计规则（v0.2 — 组件化重构版）

### 9.1 子包结构

```
antopt/
├── antenna/         # 天线基础模型
│   ├── pattern.py       — 阵因子计算器（线阵/面阵/对称/FFT快速）
│   ├── geometry.py      — Element, LinearArray, PlanarArray
│   └── element_pattern.py — 单元方向图导入（IEP/AEP/HFSS CSV）
├── mapping/         # 优化变量 → 物理位置映射
│   ├── linear.py        — LMMapper (Stick-Breaking, 仅线阵)
│   └── planar.py        — PlanarMapper (网格选通, 八邻域碰撞检测)
├── opt/             # 优化管线
│   ├── config.py        — BaseConfig → BeamformingConfig → ASMConfig
│   ├── scenario.py      — assemble_scenario（三态解析 + 默认值 + flag预计算）
│   ├── problem.py       — BeamformingProblem（六组件代价 + 变量解码）
│   └── solver.py        — CMA-ES / DE / GWO
├── analysis/        # 分析 + 可视化
│   ├── metrics.py       — PSLL / 峰值搜索 / 方向性系数积分
│   └── plotting.py      — PatternPlot / 3D / 位置图
├── io/              # 数据读写
│   ├── hfss.py          — HFSS CSV 读写 / AEP 加载
│   └── result.py        — ScenarioResult 存取 / load_elements()
└── space_mapping/   # 空间映射（ASM / MSM）
```

### 9.2 结果输出格式

**elements.txt** — 阵元数据（单一文件，替代旧的 positions/amplitudes/phases 三分立）：
```
arrayType: linear
elementsNum: 32
             x              y     amplitudes         phases
     -0.240250       0.000000       0.191999       0.000000
```

**optResult.json** — 精简结果（去掉 config_snapshot，只留核心字段）：
```json
{
  "arrayType": "linear",
  "elementsNum": 32,
  "fitness": 0.0,
  "components": {
    "sidelobe": {"psll_db": -30.4, "psll_angle_deg": -23.0, "target_db": -30},
    "main_lobe_pointing": {"target_deg": 0, "peak_deg": 0.0}
  },
  "elapsed_seconds": 1.5,
  "replay": {
    "frequenciesGHz": [9.8],
    "theta0sDeg": [0],
    "thetaDeg": [-90, 90, 0.1],
    "ep": {"csvDirectory": "...", "thetaRange": [-90, 90], "isGain": true, "inDB": true}
  }
}
```

### 9.3 角度精度规则（铁律）

所有和 `theta_step` 相关的角度值必须四舍五入到采样步长精度，不得保留浮点 noise：

```python
def _snap_angle(deg, step):
    val = round(deg / step) * step
    val = 0.0 if abs(val) < step * 0.5 else val
    ndigits = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 0
    return round(val, ndigits) if ndigits > 0 else float(val)
```

影响范围：`component_results` 中的 `psll_angle_deg`、`peak_deg`、控制台打印、图片标题和图例。

### 9.4 图片中 LaTeX 符号规范

在 matplotlib 的 f-string 中使用 `\theta_0` 时，Python 会将 `\t` 转义为 TAB 字符导致只显示 `heta`。必须使用**双反斜杠**：

```python
# ❌ 错误 — \t 被转义为 tab
title = f"$\theta_0$={t0}°"

# ✅ 正确 — \\t 是字面反斜杠 + t
title = f"$\\theta_0$={t0}°"
```

### 9.5 单元方向图 theta 范围约定

EP CSV 数据的 theta 覆盖范围默认为 `[-90°, 90°]`（非 `[-180°, 180°]`）。

- `ElementPattern.from_hfss_multi_freq()` 的 `input_theta_range` 默认值 `(-90, 90)`
- `BeamformingConfig.ep_theta_range` 默认值 `(-90, 90)`
- `Config.json` 中 `"thetaDeg"` 默认值 `[-90, 90, 0.01]`

此参数用于从 CSV 行数反算采样间隔，填错会导致 EP 插值全错位，PSLL 偏差 ~2dB。

### 9.6 画图脚本复用规则

画图脚本（`plot_linear_beam.py`、`plot_planar_beam.py`）不自行解析 `elements.txt`，统一调用库函数：

```python
from antopt.io import load_elements
arr = load_elements(result_dir)  # → np.ndarray (N, 4)
```

`load_elements` 自动处理目录/文件路径两种情况。

### 9.7 fitness 退出条件

组件模式下 `stopFitness = 0.0` — 所有激活的代价组件归零即为完美解。代价函数必须设计为恒 ≥0 的形式。

### 9.8 阵因子快速计算（FINUFFT）

#### 数学原理

阵列因子本质上是一个二维非均匀离散傅里叶变换（NDFT）：

$$AF(u,v) = \sum_{n=0}^{N-1} A_n \cdot e^{j \cdot 2\pi \cdot (x_n \cdot u + y_n \cdot v)}$$

其中 $u = \sin\theta \cdot \cos\phi$，$v = \sin\theta \cdot \sin\phi$，$x_n, y_n$ 以波长 $\lambda_0$ 为单位。

此公式对任意阵型（线/面、均匀/非均匀）完全统一。线阵退化为 $y_n = 0, v = 0$ 的特例。

#### NUFFT Type 3 加速

使用 [FINUFFT](https://github.com/flatironinstitute/finufft)（`pip install finufft`）的 Type 3 变换：

- **复杂度**：$O(N + N_{target} \cdot \log N_{target})$，vs 直接求和 $O(N \cdot N_\theta \cdot N_\phi)$
- **精度**：$\varepsilon = 10^{-6}$，实测误差 < **0.001 dB**，PSLL 差异 **0.000 dB**
- **加速比**：线阵 7-28×，面阵 20-2700×，多种群模拟 39-448×

#### 关键公式（Type 3 坐标缩放）

```python
L = max(|x|, |y|)                        # 公共归一化系数
xs, ys = x * (π / L), y * (π / L)        # 源坐标 → [-π, π]
s = (2L / 2π) * k * δu                   # 目标坐标
t = (2L / 2π) * k * δv
af = finufft.nufft2d3(xs, ys, w, s, t, eps=1e-6, isign=1)
```

推导：将 $AF = \sum w_n \cdot \exp(j \cdot k \cdot (x_n \cdot \delta u + y_n \cdot \delta v))$ 代入 FINUFFT 的 $\sum c_j \exp(i \cdot (x_j \cdot s + y_j \cdot t))$ 形式。

#### 集成方式

- `pattern.af(method="auto")` — 自动决策：finufft 可用 + N ≥ 300 → NUFFT，否则直接计算
- `pattern.af_nufft()` — 强制 NUFFT，支持多频多角度
- `_nufft_1d()` / `_nufft_2d()` — 底层封装
- Config.json 控制：`"optimizer": { "afMethod": "auto" | "direct" | "nufft" }`

#### 基准测试参考

详见 `scripts/bench_af.py` 和 `scripts/verify_nufft.py`。
