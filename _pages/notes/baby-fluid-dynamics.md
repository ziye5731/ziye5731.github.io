---
layout: archive
title: "Baby Fluid Dynamics"
permalink: /notes/baby-fluid-dynamics/
redirect_from:
  - /notes/baby-fluid-dynamics.html
author_profile: false
---

{% raw %}

这篇笔记最初根据 [Zhu Shuaichen 的流体力学短课](https://www.bilibili.com/video/BV1Ty4y1M7Fx)整理，围绕一个基本问题展开：**怎样从质量、动量和能量守恒出发，描述流体的运动？** 文中保留了原课程的示意图，并补充了推导中需要的记号与适用条件。

我们先建立流体、流体质点和控制体的概念，再依次推导三个守恒方程。重点是理解每一项对应的物理过程：某个区域内积累了多少，有多少穿过边界，以及外界带来了什么作用。

**目录**

- [1. 什么是流体](#fluid)
- [2. 用连续的场描述流体](#fields)
- [3. 跟随流体，还是观察固定空间](#descriptions)
- [4. 质量守恒：连续方程](#mass)
- [5. 动量守恒：从受力到运动](#momentum)
- [6. 伯努利方程](#bernoulli)
- [7. 能量守恒](#energy)
- [8. 从守恒方程到具体问题](#closure)

## 1. 什么是流体
{: #fluid}

区分流体与固体，一个关键是观察它们对**剪切应力**的响应。剪切应力是沿截面切向的单位面积作用力；与之相对，正应力沿截面法向作用。

在小变形、线弹性的近似下，固体的剪切应力与剪切应变成正比：

$$
\tau = G\gamma,
$$

其中，$G$ 是剪切模量，$\gamma$ 是剪切应变。施加一定的剪切应力后，固体可以在有限变形处达到平衡。

![固体在剪切力作用下发生有限变形](/images/notes/baby-fluid-dynamics/solid-shear.png)

流体则不能在静止状态下持续承受剪切应力：只要存在剪切应力，就会持续发生剪切变形。对于**牛顿流体**，剪切应力与剪切变形的速率成正比。在简单的平行剪切流中，若速度为 $u(y)$，则

$$
\tau = \mu\frac{\partial u}{\partial y},
$$

其中，$\mu$ 是动力黏度，单位为 $\mathrm{Pa\cdot s}$。这里决定应力的是速度梯度，而不是已经积累了多少变形。

![流体的剪切应力与剪切变形速率有关](/images/notes/baby-fluid-dynamics/fluid-shear.png)

静止流体仍然能够承受压强。水对容器壁的压力、气体对活塞的压力，都是例子。还要注意，**流动不一定意味着存在剪切应力**：流体整体做匀速平移时，速度处处相同，就没有由速度梯度引起的黏性应力。

## 2. 用连续的场描述流体
{: #fields}

### 2.1 连续介质假设与流体质点

微观上，流体由大量分子组成；宏观上，我们通常不追踪每一个分子，而是把流体近似看作连续分布的介质，在每个位置定义密度、速度、压强等物理量。这就是**连续介质假设**。

在这一描述下，**流体质点**是随流体一起运动的微小物质团。它相对于研究对象足够小，以至于可以把内部的宏观性质近似看作均匀；同时又包含足够多的分子，使统计平均有意义。因此，它既不是单个分子，也不是没有体积的几何点。

后文的微元与微分都在这一宏观近似下理解。固体力学与流体力学都可以采用连续介质的描述。

### 2.2 描述流体的基本物理量

**压强 $p$（pressure）**，单位为 $\mathrm{Pa}=\mathrm{N/m^2}$。对于静止流体，若面积微元 $\mathrm{d}A$ 上受到的法向压缩力大小为 $\mathrm{d}F_n$，则

$$
p = \frac{\mathrm{d}F_n}{\mathrm{d}A}.
$$

压强是单位面积上的力；压力则是力本身，两者需要区分。静止流体在同一点的压强与截面方向无关。运动的黏性流体还可能存在黏性正应力，因此截面上的总法向应力不能一概等同于压强。

![静止流体中压强的定义](/images/notes/baby-fluid-dynamics/pressure.png)

**密度 $\rho$（density）**，单位为 $\mathrm{kg/m^3}$。若体积微元 $\mathrm{d}V$ 内的质量为 $\mathrm{d}m$，则

$$
\rho = \frac{\mathrm{d}m}{\mathrm{d}V}.
$$

![流体密度的定义](/images/notes/baby-fluid-dynamics/density.png)

**温度 $T$（temperature）**，单位为 $\mathrm{K}$，描述流体的热力学状态。温度、压强与密度之间的关系由状态方程给出。例如，理想气体满足

$$
p = \rho R_s T,
$$

其中，$R_s$ 是该气体的比气体常数，单位为 $\mathrm{J/(kg\cdot K)}$。

**速度 $\boldsymbol{v}$（velocity）**，单位为 $\mathrm{m/s}$，是一个矢量。在三维直角坐标系中，记为

$$
\boldsymbol{v} = (u,v,w)^\top.
$$

这里，粗体 $\boldsymbol{v}$ 表示完整的速度矢量，$u,v,w$ 分别表示它在三个坐标方向上的分量。

这些物理量一般都随位置与时间变化。令 $\boldsymbol{x}=(x,y,z)$，我们要研究的就是

$$
p(\boldsymbol{x},t),\quad
\rho(\boldsymbol{x},t),\quad
T(\boldsymbol{x},t),\quad
\boldsymbol{v}(\boldsymbol{x},t)
$$

这些**场**如何相互联系、如何演化。

## 3. 跟随流体，还是观察固定空间
{: #descriptions}

### 3.1 拉格朗日描述：追踪同一团物质

拉格朗日描述关心的是：**同一个流体质点去了哪里，它的性质怎样变化？**

例如，在水中放入一小团染料，持续追踪这一团染料的位置，就是拉格朗日视角。研究对象始终是同一团物质，但它占据的空间可以移动、拉伸和变形。

![拉格朗日描述追踪同一组流体质点的运动](/images/notes/baby-fluid-dynamics/lagrangian-description.png)

### 3.2 欧拉描述：观察给定位置的流体

欧拉描述关心的是：**在某个空间位置，此刻的速度、密度或压强是多少？**

例如，在河道某处固定一支流速仪，记录经过该处的水流速度，就是欧拉视角。先后经过仪器的水并不是同一团物质，但观测位置保持不变。

我们还可以选取一个空间区域作为**控制体**（control volume，CV），并把它的边界称为**控制面**（control surface，CS）。流体可以穿过控制面流入或流出。

下文统一采用**固定不动的控制体** $V$，边界记为 $S=\partial V$。令 $\boldsymbol{n}$ 为控制面的外法向单位矢量，则有向面积微元为

$$
\mathrm{d}\boldsymbol{S}=\boldsymbol{n}\,\mathrm{d}S.
$$

因此，$\boldsymbol{v}\cdot\boldsymbol{n}>0$ 表示流出，$\boldsymbol{v}\cdot\boldsymbol{n}<0$ 表示流入。后文所有通量都沿用这一符号约定。若控制体本身运动，则需要改用流体相对于控制面的速度。

### 3.3 物质导数：连接两种描述

对一个标量场 $\phi(\boldsymbol{x},t)$，固定位置处的变化率是 $\partial\phi/\partial t$。但流体质点会移动，它还会经过性质不同的位置。因此，跟随该质点观察到的变化率为

$$
\frac{D\phi}{Dt}
=
\frac{\partial\phi}{\partial t}
+\boldsymbol{v}\cdot\nabla\phi.
$$

这称为**物质导数**。第一项来自场在固定位置随时间的变化，第二项来自质点穿过空间中不均匀的场。

例如，流体质点的加速度是

$$
\frac{D\boldsymbol{v}}{Dt}
=
\frac{\partial\boldsymbol{v}}{\partial t}
+(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}.
$$

因此，即使流动是定常的，即 $\partial\boldsymbol{v}/\partial t=0$，质点仍可能加速：水流通过逐渐变窄的管道时就是如此。关于这一导数与体积变化的联系，可参见 Richard Fitzpatrick 的[物质导数讲义](https://farside.ph.utexas.edu/teaching/336L/Fluid/node11.html)。

## 4. 质量守恒：连续方程
{: #mass}

质量守恒的基本思路是：

> 控制体内质量的增长率 = 质量的净流入率。

采用外法向约定后，也可以写成“积累率 + 净流出率 = 0”。

### 4.1 从边界上的质量流率出发

在很短的时间 $\Delta t$ 内，穿过面积微元 $\mathrm{d}S$ 的有符号体积约为

$$
\Delta V
\approx (\boldsymbol{v}\cdot\boldsymbol{n})\,\mathrm{d}S\,\Delta t.
$$

乘以当地密度，得到穿过该面积的质量。于是，微元上的质量流率为

$$
\mathrm{d}\dot{m}
=\rho(\boldsymbol{v}\cdot\boldsymbol{n})\,\mathrm{d}S.
$$

![由速度的法向分量计算穿过截面的体积与质量](/images/notes/baby-fluid-dynamics/mass-flux.png)

对整个控制面积分，得到质量的净流出率

$$
\Phi_m=\int_S\rho\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S.
$$

$\rho\boldsymbol{v}$ 是质量通量密度，单位为 $\mathrm{kg/(m^2\cdot s)}$；积分后的 $\Phi_m$ 是质量流率，单位为 $\mathrm{kg/s}$。

### 4.2 积分形式

控制体内的总质量为

$$
m_V(t)=\int_V\rho\,\mathrm{d}V.
$$

在没有质量源或汇的情况下，质量守恒给出

$$
\boxed{
\frac{\mathrm{d}}{\mathrm{d}t}\int_V\rho\,\mathrm{d}V
+\int_S\rho\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S
=0.
}
$$

因为控制体固定，时间导数可以移入体积分：

$$
\int_V\frac{\partial\rho}{\partial t}\,\mathrm{d}V
+\int_S\rho\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S=0.
$$

### 4.3 微分形式

利用高斯散度定理（也称奥斯特罗格拉茨基定理），

$$
\int_S\boldsymbol{a}\cdot\boldsymbol{n}\,\mathrm{d}S
=\int_V\nabla\cdot\boldsymbol{a}\,\mathrm{d}V,
$$

可将面积分改写为体积分：

$$
\int_V\left[
\frac{\partial\rho}{\partial t}
+\nabla\cdot(\rho\boldsymbol{v})
\right]\mathrm{d}V=0.
$$

由于控制体可以任意选取，在场足够光滑的区域，被积函数必须处处为零。因此得到**连续方程**：

$$
\boxed{
\frac{\partial\rho}{\partial t}
+\nabla\cdot(\rho\boldsymbol{v})=0.
}
$$

其中，$\nabla$ 是微分算子，在直角坐标系中可写为

$$
\nabla=\left(
\frac{\partial}{\partial x},
\frac{\partial}{\partial y},
\frac{\partial}{\partial z}
\right)^\top.
$$

对矢量场使用 $\nabla\cdot$ 得到散度。连续方程展开后为

$$
\frac{\partial\rho}{\partial t}
+\frac{\partial(\rho u)}{\partial x}
+\frac{\partial(\rho v)}{\partial y}
+\frac{\partial(\rho w)}{\partial z}=0.
$$

这一积分到微分的推导也可参见[质量守恒讲义](https://farside.ph.utexas.edu/teaching/336L/Fluid/node10.html)。

### 4.4 定常流动与不可压流动

**定常流动**指各场量在固定位置不随时间变化。此时 $\partial\rho/\partial t=0$，连续方程化为

$$
\nabla\cdot(\rho\boldsymbol{v})=0,
\qquad
\int_S\rho\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S=0.
$$

也就是说，控制体的质量净流出率为零。流入和流出都可以非零，只要二者相等。

**不可压流动**指流体质点的体积在运动中保持不变。对于质量不变的质点，这等价于密度沿其运动轨迹保持不变，即 $D\rho/Dt=0$。将连续方程展开为

$$
\frac{D\rho}{Dt}+\rho\nabla\cdot\boldsymbol{v}=0,
$$

便得到

$$
\boxed{\nabla\cdot\boldsymbol{v}=0.}
$$

这里不必要求密度在整个空间处处相同；**均匀、恒定的密度是更强的假设**，也是入门问题中常用的情形。不可压与定常是不同的条件，二者不能互相替代。

速度散度还有直接的几何意义。对随流体运动的体积微元 $\delta V$，

$$
\frac{1}{\delta V}\frac{D(\delta V)}{Dt}
=\nabla\cdot\boldsymbol{v}.
$$

散度为正表示局部膨胀，为负表示局部压缩，为零表示体积不变；体积不变并不妨碍形状发生剪切变形。

## 5. 动量守恒：从受力到运动
{: #momentum}

动量方程是牛顿第二定律在流体中的表达。对于固定控制体，它的结构是：

> 控制体内动量的增长率 + 动量的净流出率 = 外界作用的合力。

### 5.1 流体受到哪些力

**体积力**作用于流体内部，例如重力。记 $\boldsymbol{f}$ 为单位质量流体受到的体积力，单位为 $\mathrm{m/s^2}$，则控制体受到的体积力为

$$
\boldsymbol{F}_{\mathrm{body}}
=\int_V\rho\boldsymbol{f}\,\mathrm{d}V.
$$

若只考虑重力，$\boldsymbol{f}=\boldsymbol{g}$。

**表面力**通过控制面作用于流体。它既包括压强产生的力，也包括黏性应力产生的力。用应力张量统一表示为

$$
\boldsymbol{\sigma}=-p\boldsymbol{I}+\boldsymbol{\tau},
$$

其中，$\boldsymbol{I}$ 是单位张量，$\boldsymbol{\tau}$ 是黏性应力张量。对于外法向为 $\boldsymbol{n}$ 的面积微元，表面力为

$$
\mathrm{d}\boldsymbol{F}_{\mathrm{surface}}
=\boldsymbol{\sigma}\cdot\boldsymbol{n}\,\mathrm{d}S
=\left(-p\boldsymbol{n}
+\boldsymbol{\tau}\cdot\boldsymbol{n}\right)\mathrm{d}S.
$$

压强项前的负号来自方向约定：$\boldsymbol{n}$ 朝控制体外侧，而压强施加的力朝内侧。黏性应力则可能同时包含法向和切向分量。

### 5.2 积分形式

控制体内的总动量为 $\int_V\rho\boldsymbol{v}\,\mathrm{d}V$。穿过控制面的每一份质量还携带着当地的速度，因此动量的净流出率为

$$
\int_S\rho\boldsymbol{v}
(\boldsymbol{v}\cdot\boldsymbol{n})\,\mathrm{d}S.
$$

于是动量守恒可写为

$$
\begin{aligned}
&\frac{\mathrm{d}}{\mathrm{d}t}
\int_V\rho\boldsymbol{v}\,\mathrm{d}V
+\int_S\rho\boldsymbol{v}
(\boldsymbol{v}\cdot\boldsymbol{n})\,\mathrm{d}S\\
&\qquad=
-\int_S p\boldsymbol{n}\,\mathrm{d}S
+\int_S\boldsymbol{\tau}\cdot\boldsymbol{n}\,\mathrm{d}S
+\int_V\rho\boldsymbol{f}\,\mathrm{d}V.
\end{aligned}
$$

三项外力分别是压强力、黏性力和体积力。每一项的单位都是力的单位 $\mathrm{N}$。

### 5.3 从积分形式到局部方程

先看 $x$ 方向的动量。它的密度是 $\rho u$，通量密度是矢量 $\rho u\boldsymbol{v}$。利用散度定理，可得

$$
\frac{\partial(\rho u)}{\partial t}
+\nabla\cdot(\rho u\boldsymbol{v})
=-\frac{\partial p}{\partial x}
+(\nabla\cdot\boldsymbol{\tau})_x
+\rho f_x.
$$

这里应当使用**散度** $\nabla\cdot(\rho u\boldsymbol{v})$。面积分转成体积分后，积分微元也应为 $\mathrm{d}V$。

其中，黏性力的局部表达是应力张量的散度。例如，在直角坐标系中，

$$
(\nabla\cdot\boldsymbol{\tau})_x
=\frac{\partial\tau_{xx}}{\partial x}
+\frac{\partial\tau_{xy}}{\partial y}
+\frac{\partial\tau_{xz}}{\partial z}.
$$

它是单位体积受到的黏性力，单位为 $\mathrm{N/m^3}$；对控制体积分后，才得到总黏性力。

接下来展开左边，利用连续方程消去与质量变化有关的项：

$$
\begin{aligned}
\frac{\partial(\rho u)}{\partial t}
+\nabla\cdot(\rho u\boldsymbol{v})
&=\rho\frac{Du}{Dt}
+u\left[
\frac{\partial\rho}{\partial t}
+\nabla\cdot(\rho\boldsymbol{v})
\right]\\
&=\rho\frac{Du}{Dt}.
\end{aligned}
$$

其余两个方向同理。合并后得到

$$
\boxed{
\rho\frac{D\boldsymbol{v}}{Dt}
=-\nabla p
+\nabla\cdot\boldsymbol{\tau}
+\rho\boldsymbol{f}.
}
$$

其中，$\nabla p$ 是压强的**梯度**，指向压强增加最快的方向；所以 $-\nabla p$ 对应从高压指向低压的力密度。上述推导可对照[动量守恒讲义](https://farside.ph.utexas.edu/teaching/336L/Fluid/node12.html)。

### 5.4 欧拉方程与 Navier–Stokes 方程

若忽略黏性应力，即 $\boldsymbol{\tau}=0$，得到**欧拉动量方程**：

$$
\rho\left[
\frac{\partial\boldsymbol{v}}{\partial t}
+(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}
\right]
=-\nabla p+\rho\boldsymbol{f}.
$$

若考虑牛顿流体，则还要用应力与变形速率之间的本构关系，把 $\boldsymbol{\tau}$ 写成速度梯度的函数，由此得到 **Navier–Stokes 方程**。

对于不可压牛顿流体，

$$
\boldsymbol{\tau}
=\mu\left[
\nabla\boldsymbol{v}
+(\nabla\boldsymbol{v})^\top
\right],
$$

这里采用 $(\nabla\boldsymbol{v})_{ij}=\partial v_i/\partial x_j$ 的记号。若进一步假设 $\mu$ 为常数，则

$$
\nabla\cdot\boldsymbol{\tau}
=\mu\nabla^2\boldsymbol{v}.
$$

对于密度也均匀、恒定的情形，便得到常见的不可压 Navier–Stokes 方程组：

$$
\boxed{
\begin{aligned}
\nabla\cdot\boldsymbol{v}&=0,\\
\frac{\partial\boldsymbol{v}}{\partial t}
+(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}
&=-\frac{1}{\rho}\nabla p
+\nu\nabla^2\boldsymbol{v}
+\boldsymbol{f},
\end{aligned}
}
$$

其中，$\nu=\mu/\rho$ 是运动黏度，单位为 $\mathrm{m^2/s}$。这个简洁形式依赖于牛顿流体、不可压和黏度恒定等假设，不能直接套用于任意黏性流动。更多说明见[不可压流动方程讲义](https://farside.ph.utexas.edu/teaching/336L/Fluid/node15.html)。

## 6. 伯努利方程
{: #bernoulli}

伯努利方程可以看作欧拉动量方程沿流线积分的结果。考虑以下条件：

- 流动定常；
- 黏性效应可以忽略；
- 密度 $\rho$ 为常数；
- 体积力只有均匀重力；
- 比较的两点位于同一条流线上。

取 $z$ 轴竖直向上，沿流线的微小位移记为 $\mathrm{d}s$，速率记为 $U=\lVert\boldsymbol{v}\rVert$。将欧拉方程投影到流线方向，可得

$$
U\frac{\mathrm{d}U}{\mathrm{d}s}
=-\frac{1}{\rho}\frac{\mathrm{d}p}{\mathrm{d}s}
-g\frac{\mathrm{d}z}{\mathrm{d}s}.
$$

沿流线积分后，得到

$$
\boxed{
\frac{p}{\rho}+\frac{U^2}{2}+gz
=\text{沿同一条流线的常数}.
}
$$

若忽略重力，或比较同一高度的两点，则简化为

$$
p_1+\frac{1}{2}\rho U_1^2
=p_2+\frac{1}{2}\rho U_2^2.
$$

因此，“流速越大，压强越小”需要放在这些条件下理解。一般情况下，不同流线上的伯努利常数可以不同；若流动还满足无旋等附加条件，才可进一步在相应的连通流动区域内使用同一个常数。推导与适用范围可参见 MIT 的[伯努利方程讲义](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec08/lecture8.html)。

## 7. 能量守恒
{: #energy}

能量方程同样遵循“积累 + 净流出 = 外界输入”的结构，但需要区分两种机制：**流体穿过边界时携带能量，以及热传递和外力做功改变能量**。

### 7.1 控制体内储存了什么能量

记 $e$ 为单位质量的内能，定义

$$
E=e+\frac{1}{2}\lVert\boldsymbol{v}\rVert^2.
$$

$E$ 是单位质量的内能与动能之和，单位为 $\mathrm{J/kg}$。这里暂不把重力势能计入 $E$，而将重力的作用记在体积力做功中，以避免重复计算。

控制体内的能量为 $\int_V\rho E\,\mathrm{d}V$，随流体运动而净流出的能量流率为

$$
\int_S\rho E\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S.
$$

### 7.2 热传递与外力做功

用 $\boldsymbol{q}$ 表示导热热流密度，单位为 $\mathrm{W/m^2}$。其方向取能量传递的方向，因此通过控制面输入的净热流率是

$$
-\int_S\boldsymbol{q}\cdot\boldsymbol{n}\,\mathrm{d}S.
$$

对于各向同性介质，傅里叶定律给出 $\boldsymbol{q}=-k\nabla T$，其中 $k$ 为导热系数。负号表示热量由高温处传向低温处。

此外，用 $\dot{q}_{\mathrm{vol}}$ 表示单位体积的热源功率，单位为 $\mathrm{W/m^3}$。例如，在采用局部热源近似时，可以把辐射吸收的贡献计入这一项。

外力做功则分为两类。体积力的功率为

$$
\int_V\rho\boldsymbol{f}\cdot\boldsymbol{v}\,\mathrm{d}V;
$$

表面力的功率为

$$
\begin{aligned}
\int_S(\boldsymbol{\sigma}\cdot\boldsymbol{n})
\cdot\boldsymbol{v}\,\mathrm{d}S
&=-\int_S p\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S\\
&\quad+\int_S(\boldsymbol{\tau}\cdot\boldsymbol{n})
\cdot\boldsymbol{v}\,\mathrm{d}S.
\end{aligned}
$$

右边分别是压强力做功和黏性应力做功。它们与能量随质量流动穿过边界的过程不同，需要分别计入。

### 7.3 积分形式

把上述各项合在一起，得到

$$
\begin{aligned}
&\frac{\mathrm{d}}{\mathrm{d}t}
\int_V\rho E\,\mathrm{d}V
+\int_S\rho E\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S\\
&\quad=-\int_S p\boldsymbol{v}\cdot\boldsymbol{n}\,\mathrm{d}S
+\int_S(\boldsymbol{\tau}\cdot\boldsymbol{n})
\cdot\boldsymbol{v}\,\mathrm{d}S\\
&\qquad\quad+\int_V\rho\boldsymbol{f}\cdot\boldsymbol{v}\,\mathrm{d}V
-\int_S\boldsymbol{q}\cdot\boldsymbol{n}\,\mathrm{d}S
+\int_V\dot{q}_{\mathrm{vol}}\,\mathrm{d}V.
\end{aligned}
$$

左边依次是储能变化率与能量净流出率；右边依次是压强功、黏性功、体积力功、导热输入和体积热源。所有项的单位都是 $\mathrm{W}$。

### 7.4 微分形式

对于应力张量对称的普通流体，将表面功与热流项用散度定理改写，再利用控制体的任意性，可得局部能量方程：

$$
\boxed{
\begin{aligned}
\frac{\partial(\rho E)}{\partial t}
+\nabla\cdot(\rho E\boldsymbol{v})
&=-\nabla\cdot(p\boldsymbol{v})
+\nabla\cdot(\boldsymbol{\tau}\cdot\boldsymbol{v})\\
&\quad+\rho\boldsymbol{f}\cdot\boldsymbol{v}
-\nabla\cdot\boldsymbol{q}
+\dot{q}_{\mathrm{vol}}.
\end{aligned}
}
$$

微分方程中的每一项都是单位体积的功率，单位为 $\mathrm{W/m^3}$。因此，控制体上的总热流率或总功率不能直接搬到局部方程中，必须先写成相应的通量散度或体积源项。

如果再减去由动量方程得到的动能方程，还可得到内能形式：

$$
\rho\frac{De}{Dt}
=-p\nabla\cdot\boldsymbol{v}
+\boldsymbol{\tau}:\nabla\boldsymbol{v}
-\nabla\cdot\boldsymbol{q}
+\dot{q}_{\mathrm{vol}}.
$$

其中，$\boldsymbol{\tau}:\nabla\boldsymbol{v}
=\sum_{i,j}\tau_{ij}\,\partial v_i/\partial x_j$ 是应力与速度梯度的双重缩并。对于通常的牛顿流体，这一项表示黏性耗散，描述机械能向内能的转化；总能量仍然守恒。能量方程及其内能形式可对照[能量守恒讲义](https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node14.html)。

## 8. 从守恒方程到具体问题
{: #closure}

质量、动量与能量守恒给出了描述流体的基本约束。要真正求解一个问题，还需要补充物质的性质与具体的物理条件：

- **本构关系**：例如，黏性应力与速度梯度之间的关系，以及热流与温度梯度之间的关系。
- **热力学关系**：例如，理想气体状态方程，以及内能与温度等状态量之间的关系。
- **初始条件与边界条件**：例如，初始速度分布、入口流量、壁面条件和边界温度。

在密度、黏度均取常数，且温度不反过来影响流动的模型中，可以先由不可压连续方程与动量方程求速度和压强；若还关心温度变化，再求解相应的热能方程。

面对具体流动，先说明控制体如何选取、哪些物理作用需要保留、哪些近似可以成立，再写下相应的方程。这样，每一个被保留或略去的项都有清楚的物理依据。

{% endraw %}
