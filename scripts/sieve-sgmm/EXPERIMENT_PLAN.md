# Streaming Sieve GMM：实验设计与所需结果

这份方案对应博客 `Streaming Sieve GMM for Nonlinear Instrumental Variable Regression` 的两项理论结果：有限基函数的 identification 条件，以及固定基函数下的均方收敛速率。当前工作只完成正文与实验设计，不继续执行实验。本文件属于研究与复现材料，由网站现有 `scripts` 排除规则保留在仓库中，不进入网页。

## 1. 实验需要回答的具体问题

实验的叙述顺序应与理论顺序一致：

1. **Identification：**有效工具变量是否被所选基函数捕获？矩数量足够是否仍可能缺少识别方向？局部满秩为何不等于全局唯一？
2. **Convergence：**在能核验定理全部条件的设置里，参数均方误差是否与上界相容？识别强度、固定步长系数与 Jacobian tracking 分别如何影响有限样本表现？
3. **复杂模型中的适用范围：**高阶基函数新增的是识别信号还是主要新增噪声？哪些现象能够由上述分析解释，哪些还需要单独的稳定性或全局识别分析？

不把“曲线下降”当作 identification 证明，不把某个拟合斜率当作速率定理的证明，也不只用中位数评价控制期望的定理。

## 2. 优先级与执行规模

首轮必须完成 A、B、D；C 用于进一步拆解速率上界；E 是正文扩展时的非线性压力测试。

- **A：基函数与 identification 的解析对照。**一维非线性模型，degree 1 与 degree 2 对照；同时计算总体矩函数与解析信号强度。
- **B：均方速率与识别强度。**复用 A 的模型，固定 harmonic 系数比较强、弱信号，再调整系数满足定理阈值。
- **C：tracking 与步长分离。**改变两种指数，记录 parameter error 和 Jacobian tracking error；加入 population-Jacobian 诊断基准。
- **D：多维二次结构与基函数扩展。**先用线性 first stage 建立解析 rank 对照，再用 ReLU first stage 检验新增高阶信号。
- **E：exp–tanh 非线性参数压力测试。**分析初始化、局部区域、曲率及跨运行尾部误差，不作未经验证的全局速率结论。

建议 A–C 使用 512 次独立重复，每次 `T=2^18=262144` 个训练观测，主随机种子 `20261023`，预先固定 97 个对数等距检查点。D、E 先设计 64 次重复、每次 `T=2^18`；若尚未达到稳定区间，只按预先规定的倍增序列扩展到 `2^19`、`2^20`，并报告所有预算，避免根据结果选择展示区间。不得为得到目标斜率反复更换种子或选择重复运行。

可在开始正式运行前进行一次单独的代码正确性检查；其随机种子不进入正式样本，不据此挑选有利的模型或基函数。正式运行前保存完整配置、代码 commit 和依赖版本。

## 3. A–C 的共同模型：全部理论量可解析

设每个时刻的 `z,u,nu` 相互独立且均为标准正态；不同时间、不同重复相互独立。令

$$
x=\gamma z+u,\qquad
\varepsilon=0.5u+0.5(u^2-1)+\nu,\qquad
y=g(\theta_*;x)+\varepsilon,
$$

$$
g(\theta;x)=(\theta+0.01\theta^2)x^2,\qquad \theta_*=0.
$$

这个模型既非线性于 regressor，也非线性于 parameter。工具变量有效，因为 `E[epsilon|z]=0`；regressor 与结构误差相关，因为 `Cov(x,epsilon)=0.5`。此外 `E[x^2*epsilon]=1`，所以非线性结构特征本身也与误差相关，普通结构最小二乘在真值处的总体 score 不为零。选择零真值是为了使总体矩和区域条件透明，并不意味着算法获得真值；该组实验是定理条件校准，不代表实际应用中可以围绕未知真值构造区域。

统一使用：

- `Theta=[-1,1]`，每一步投影，`theta_0=0.5`，`S_0=0`。
- `lambda=1`，`t0=100`，`beta_t=(t+100)^(-2/3)`。
- `alpha_t=a/(t+100)`，除 C 的指数消融外不变。
- 先用旧 `S_t` 与旧 `theta_t` 更新参数，再用同一新观测在旧参数处更新 `S`。
- 同时报最后迭代与从第一次更新开始的等权平均；不事后丢弃 burn-in。
- 该处的“弱信号”是两个固定正识别强度的有限样本比较，不是随样本量变化的 weak-IV 渐近序列。

### 3.1 可直接核验的 identification

记 `f(theta)=theta+0.01 theta^2`，则

$$
b_\theta(z)=f(\theta)(1+\gamma^2z^2).
$$

- 对 `psi_1=z`，总体矩对所有参数均为零。因此 `G_1=0`、`mu_1=0`，该基函数不识别参数。
- 对 `psi_2=(z^2-1)/sqrt(2)`，

$$
\bar m_2(\theta)=\sqrt2\gamma^2f(\theta),\quad
\bar M_2(\theta)=\sqrt2\gamma^2(1+0.02\theta),\quad
\mu_2=2\gamma^4.
$$

`f` 在 `[-1,1]` 上严格递增且仅在零点为零，因此 degree 2 在该区域内全局识别。degree 1 的不识别不依赖优化器表现。

该对照只有一个参数、一条矩，排除了“只是因为矩数量太少”的解释；也说明不能从 `p>=d` 直接推断 identification。

### 3.2 可直接核验的局部区域

对 degree 2，`G=sqrt(2)*gamma^2`，`L=0.02*G`。与正文同样定义

$$
h=\frac{2\gamma^4}{2\gamma^4+1},\qquad
c=0.02G\left(\frac54G+\frac14\right).
$$

- `gamma=1`：`h=2/3`，`c≈0.05707`，满足 `r*c<=h/4`，其中 `r=1`。
- `gamma=0.5`：`h=1/9`，`c≈0.004893`，同样满足该区域条件。

模型及矩是高斯变量的多项式，参数集有界，因此所需一致四阶矩有限。所选 offset 保证 `beta_t<=1`、`h*alpha_t<=1`。这组实验不需要把经验上“看起来稳定”当作局部区域条件的替代。

## 4. A：缺失识别方向

固定 `gamma=1,a=3`，只改变 degree 1 与 degree 2。

需要保存并展示：

1. `theta` 网格上的解析总体矩曲线，清楚显示 degree 1 的整条零矩与 degree 2 的唯一零点。
2. 两组的 `E_hat[(theta_t-theta_star)^2]` 与平均参数的对应均方误差。
3. 末次参数分布、中位数、均值及 90% 分位数；参数误差图与总体矩图对应。
4. 投影触发率与 Jacobian 状态，以排查边界效应。

分析要点：degree 1 未恢复真值符合 population identification 的失败；degree 2 的恢复必须与解析 rank、区域条件和步长条件共同解释。若不识别组某次偶然靠近零，不能据此判断其已被识别。

## 5. B：速率、信号强度与 harmonic 系数

采用 degree 2，预先规定三个主配置：

- `gamma=1,a=3`：`a*h=2>1`，定理给出最后迭代和平均迭代均为 `O(1/T)`。
- `gamma=0.5,a=3`：`a*h=1/3`，在相同 ridge 和系数下只得到 `O(T^(-1/3))` 的保守上界。
- `gamma=0.5,a=12`：`a*h=4/3>1`，恢复 `O(1/T)` 的充分条件，但常数、瞬态和投影行为可能改变。

可选边界配置为 `gamma=0.5,a=9`，对应 `a*h=1` 与 `O(log(T)/T)` 上界。这个边界通常难以在有限样本中可靠区分，不应作为首轮必需的核心图。

### 所需结果

- 最后迭代和平均迭代的经验 MSE 双对数曲线，各自标注 `T^-1` 参照线。
- `T*MSE(T)` 曲线，用于观察 `1/T` 标度下的有限样本变化；不要求趋于常数，因为定理只给上界。
- 终点 MSE、Monte Carlo standard error、RMSE、中位数误差、90% 分位数和投影触发率。
- 用预先规定的末段 `T/4<=t<=T` 检查点拟合描述性 log–log 斜率。若报告斜率区间，应按整条重复轨迹做 bootstrap，不能把相关时间点当独立样本。

### 分析边界

定理上界不是 sharp lower bound。“弱信号、固定系数”的经验下降快于 `T^-1/3` 完全可能，不能视为反例；也不能为了匹配该保守指数改变展示区间。重点是检验信号减弱是否延长瞬态，以及调整系数后是否与 `1/T` 上界相容。不能把平均轨迹稳定直接解释为更高统计效率。

## 6. C：tracking 与两种步长

固定 degree 2、`gamma=1`，比较以下指数：

- `(eta,kappa)=(0.7,0.6)`：一般非线性参数上界的 `rho=0.4`。
- `(eta,kappa)=(0.9,0.6)`：`rho=0.9`。
- `(eta,kappa)=(1,2/3)` 且 `a=3`：满足 harmonic 的 `1/T` 条件。

各组用相同 `a=3,b=1,t0=100,lambda=1`，但应说明实际早期步长不同。另增加 **population-Jacobian 诊断基准**：将参数增益中的 `S_t` 替换为可解析的 `Mbar(theta_t)`，其余观测与设置相同。该基准利用已知总体信息，只用来隔离 tracking 的影响，不作为同等信息条件下的可部署竞争算法。

必须记录：

$$
\delta_t=S_t-\bar M(\theta_t),\qquad
\widehat{\mathbb E}\|\delta_t\|^2,
\qquad
\widehat{\mathbb E}\|\delta_t\|^4.
$$

图中将 tracking 四阶矩与 `t^(-2*kappa)+t^(-4*(eta-kappa))` 的标度参照对应，同时展示参数 MSE。这个多项式模型的 tracking 上界可能明显保守，应据实解释，不能声称两项上界分别必然主导。

补充一个 `g(theta;x)=theta*x^2` 的参数线性对照。其 `L=0`、`Mbar` 常数，应去掉 moving-target 项，理论给 `eta<1` 时 `O(T^-eta)`。它用于验证正文对“非线性于 x”与“非线性于 theta”的区分。

## 7. D：多维二次结构与基函数阶数

先设 `d_x=4,d_z=8`，令 `Gamma` 为满列秩的 `8x4` 固定矩阵，所有重复共享。使用

$$
z\sim N(0,I_8),\quad
c\sim N(0,0.5I_4),\quad
v\sim N(0,0.5I_4),\quad e\sim N(0,1),
$$

各基础变量相互独立，并设

$$
x=\phi(\Gamma^\top z)+c+v,\qquad
\varepsilon=\tfrac12\mathbf 1^\top c+e,
$$

$$
g(\theta;x)=b^\top x+x^\top A x,\qquad A=A^\top.
$$

参数包括 `b` 和 `A` 的上三角，共 14 维。统一说明上三角参数化中非对角项的梯度是 `2*x_i*x_j`，避免范数/梯度约定混淆。可用主种子 10 生成固定 `Gamma_ij~N(0,1)`、`b_star~N(0,4I_4)`、`A_star=I_4`，并把实际矩阵和向量保存成配置数据，不只写种子。初始化 `b_0~N(0,0.01I_4)`、`A_0=0`；记录每个重复的初始化种子。投影球半径 20，运行前核验真值位于球内。

比较 `K={1},{0,1},{0,1,2},{0,1,2,3}`，矩维度分别为 8、9、45、165。

### D1：线性 first stage，提供解析 rank 对照

取 `phi(s)=s`。因模型线性于参数，`L=0`，不需要局部曲率半径条件。对参数扰动 `(Delta b,Delta A)`，

$$
b_\theta(z)=\Delta b^\top\Gamma^\top z
  +z^\top\Gamma\Delta A\Gamma^\top z
  +\operatorname{tr}(\Delta A).
$$

因此：

- degree 1 只检测线性参数方向，解析 rank 为 4；`p<14` 也已排除 full column rank。
- 增加 constant 只增加一个独立二次方向，解析 rank 为 5；仍有 `p=9<14`，不能识别全部二次系数。
- 加入完整 degree 2，在 `Gamma` 满列秩时可恢复全部对称二次方向，从而获得 rank 14。
- degree 3 的总体 Jacobian block 为零，不新增局部信号。

需要输出总体 `G` 的 rank、最小奇异值、`mu`、MSE 和 structural risk，作为“先检查 signal，再讨论优化”的核心多维例子。无信息高阶矩可能增加噪声，但不能预设其性能一定变差。

### D2：ReLU first stage，检查高阶矩是否新增信号

取 `phi(s)=max(s,0)`，其余设定保持一致。结构仍线性于参数，`L=0`；变化来自 conditional sensitivity 关于 `z` 的展开。先用**独立 pilot 数据**估计 degree blocks 的总体 Jacobian、最小奇异值及数值误差，再与收敛结果对应。建议 pilot `2^20` 个样本、分批累积，至少用两个独立 pilot 种子复核。

有限 pilot 的非零最小奇异值不能直接当作总体 full rank 的证明；报告 batch-mean Monte Carlo 误差，并把数值证据与 D1 的解析结果分开。特别计算新增 degree 3 block 的大小，避免仅凭最终误差较小就宣称“高阶基函数改善 identification”。

### 共同优化设置与图表

为直接对应定理，可先用 `eta=0.9,kappa=0.6,a=0.1,b=1,t0=100,lambda=0.01`：参数线性模型的上界是 `O(T^-0.9)`，不需要 harmonic 系数阈值。若另做 harmonic 实验，须先根据实际 `mu` 选择满足 `a*h>1` 的系数，并披露其变化。

所需结果为：一张各基函数的 signal/rank 表，一张参数 MSE/structural risk 曲线图，一张终点误差与计算时间表。解释 degree 2 是否恢复原先缺失的方向，以及 degree 3 在两个 first stage 中为何具有不同作用。

## 8. E：exp–tanh 的非线性压力测试

保持 `d_x=4,d_z=8` 和 D 的基础噪声结构，但取

$$
x=\tanh(\Gamma^\top z)+c+v,\quad
\Gamma_{ij}\sim N(0,0.1^2),\quad
\varepsilon=0.25\mathbf1^\top c+0.1e,
$$

$$
g(\theta;x)=\exp(\theta^\top x).
$$

用固定、独立的 pilot 确定 `theta_star` 的尺度，使 `Var(theta_star^T x)=0.25`，保存最终向量与 pilot 精度；训练样本不参与这项定标。这个设置对应原有 exp–tanh 实验的主要尺度。

这组实验应比较近真值的局部初始化与统一的零附近初始化，并记录每条轨迹离真值的最大距离。使用局部投影的组必须明确其半径，并解释是否已验证区域不等式；没有验证时只能报告数值表现，不能将其称为定理覆盖的运行。

需要重点展示跨运行尾部：经验 MSE、均值、中位数、90% 分位数、每条终点误差。若指数计算加入 clipping、方向截断、重启或早停，应单独作为实现变体报告；它们改变结构函数或算法，不应默认为正文递推的同一实验。

本组用于识别曲率、初始化和额外矩的有限样本影响，不预设 degree 2 或 degree 3 必须胜出。

## 9. 统一统计口径

每个检查点 `t`、每种估计器 `theta_hat` 都保存逐重复的参数和误差。主要指标是

$$
\widehat{\mathrm{MSE}}_\theta(t)
  =R^{-1}\sum_{r=1}^R\|\hat\theta_t^{(r)}-\theta_*\|^2.
$$

Monte Carlo standard error 使用每次重复的**平方误差**计算：

$$
\mathrm{MCSE}(t)=\operatorname{SD}
   (\|\hat\theta_t^{(r)}-\theta_*\|^2)/\sqrt R.
$$

如展示 `mean +/-1.96*MCSE`，标明它是基于重复样本的正态近似、逐点 Monte Carlo 区间，不是所有时刻同时成立的置信带。若分布明显有长尾，同时提供逐重复数据和按重复 bootstrap 的结果，不用中位数替代均方定理对应的指标。

第二主要指标是独立测试集上的 **structural risk**：

$$
R^{-1}\sum_r\frac1{n_{test}}\sum_j
  [g(\hat\theta_t^{(r)};x_j)-g(\theta_*;x_j)]^2.
$$

测试集独立于训练流和 pilot；固定测试数据可用于配对比较，建议 `n_test=2^15`。说明测试积分误差与训练重复误差的区别。

含 outcome 的 `E[(g(theta;x)-y)^2]` 仅作辅助指标。由于内生性，其展开含有

$$
-2\mathbb E[(g(\theta;x)-g(\theta_*;x))\varepsilon],
$$

该项一般不为零。因此 `E[epsilon^2]` 不是对所有候选参数都成立的预测误差下界，低 outcome MSE 也不直接表示正确的结构参数恢复。

计算预算同时报告：训练观测数、pilot 观测数、测试集规模、运行时间、moment dimension 和参数维度。正式方法使用每步一个普通观测，vectorization 只能跨独立重复，不能把每步改成未说明的 minibatch。

## 10. 最终实验部分需要交付的材料

最小正文实验包为：

1. **设置说明。**完整 DGP、固定真值、基函数、投影、步长、ridge、初始化、重复数、种子、评价指标与训练预算。
2. **Identification 图。**A 的总体矩曲线与参数恢复对照；D1 的解析 rank / signal 表。
3. **速率图。**B 的最后迭代和平均迭代 MSE、`T*MSE` 与预先定义的末段统计；全部图注说明 MC 不确定性。
4. **基函数作用图。**D1/D2 的 MSE 与 structural risk，同时给出新增矩的 signal 诊断。
5. **结果分析。**逐条回到理论：缺失方向、信号阈值、tracking、曲率与噪声；标明定理覆盖的设置和仅有数值证据的设置。
6. **复现包。**单入口运行脚本、配置 JSON、环境版本、代码 commit、逐重复检查点数据、汇总 CSV、绘图脚本与图。图表必须从保存的数据生成，不能手工录入数值。

C 的 tracking 图和 E 的多初始化分布图可放进补充材料；若篇幅允许，再选对正文结论确有帮助的部分。

## 11. 旧实验的使用边界

已经核对 RA 仓库 `results/2-nonlinear-comparative/` 下的结果、配置和日志；不继续运行或修改该项目。

- `quadratic-linear` 的实际终点是 `5,000,000` 次迭代，保存的参数步长为 `0.01*t^-0.5`。不能与另外两组一起描述成统一的 `10^7` 次迭代、`eta=0.7,kappa=0.6` 设置，也不能用来检验正文的 harmonic 速率。
- `quadratic-relu` 与 `exp-tanh` 保存了 `10,000,000` 次迭代、10 次重复的数据。归档配置包含 `eta=0.7,kappa=0.6`；仅凭配置文件不能确认每项历史实现细节，正式引用前还应核对执行时的代码版本、投影与报告的是最后迭代还是平均迭代。
- `exp-tanh` 中，degree 2 的终点中位数误差约为 `0.00948`，但平均误差约为 `0.1676`。二者差异说明只展示中位数会隐藏部分运行的较大误差。归档 TOSG 的均值与中位数差异更大，不能只摘录表现较好的统计量。
- 如使用旧图，必须给出 median/IQR 的实际定义，不能把它们称作 mean-square convergence 图。若从 `mean` 和总体标准差字段恢复平方误差均值，应先核对归档聚合代码的 `ddof=0` 约定，再用 `mean^2+std^2`。
- TOSG 使用不同的 conditional two-sample oracle。若保留为额外数值背景，应单列 oracle 条件、完整训练抽样成本与稳定化处理；不能把等迭代数当作等样本或等信息预算下的比较。

旧结果可用于提出上述问题与选择压力测试，但正式实验叙述以可版本化、可核验、与理论一致的新设置为准。新实验重新获得授权后，再按本方案执行。
