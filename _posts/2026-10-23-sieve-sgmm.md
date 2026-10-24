---
title: "Streaming Sieve GMM for Nonlinear Instrumental Variable Regression"
date: 2026-10-23
permalink: /posts/2026/10/sieve-sgmm/
excerpt: "One-observation streaming GMM for nonlinear IV regression, with explicit conditions for finite-basis identification and finite-time mean-square convergence."
layout: single
author_profile: false
read_time: true
comments: false
share: false
related: false
categories:
  - blog
tags:
  - instrumental variables
  - stochastic approximation
  - generalized method of moments
  - sieve methods
---

An instrumental variable can be valid while a particular collection of
instrument moments fails to identify the structural parameter. Even when
those moments identify the parameter, a nonlinear online algorithm needs
additional conditions to reach it. These are different requirements, and a
streaming method should make both explicit.

This post develops **streaming sieve generalized method of moments
(SSGMM)** for a correctly specified, finite-dimensional nonlinear IV model.
A finite collection of instrument functions converts a conditional
restriction into unconditional moments. A running Jacobian then supplies a
damped Gauss–Newton direction using one new observation per update. The
theory has two parts: a basis-dependent condition for regular local
identification, and a finite-time mean-square convergence bound. The latter
includes an $O(T^{-1})$ setting, with explicit restrictions on the step
sizes, identification strength, and parameter region.

The construction combines the moment formulation of
[Hansen (1982)][hansen], the nonlinear SGMM direction discussed by
[Chen et al. (2025, Section 7)][sgmm], and two-time-scale stochastic
approximation [Borkar (1997)][borkar]. The rank condition is classical;
the analysis below expresses its consequences for the selected instrument
basis and for this particular streaming recursion. All results keep the
basis and parameter dimensions fixed as the stream grows.

## Contents

- [Section 1: Model and finite-basis moments](#section-1-model-and-finite-basis-moments)
- [Section 2: A predictable Jacobian gives a one-observation update](#section-2-a-predictable-jacobian-gives-a-one-observation-update)
- [Section 3: Identification: which parameter directions do the moments see?](#section-3-identification-which-parameter-directions-do-the-moments-see)
- [Section 4: Convergence rates: identification, curvature, and tracking](#section-4-convergence-rates-identification-curvature-and-tracking)
- [Section 5: Experiments: identification, rates, and finite-sample limitations](#section-5-experiments-identification-rates-and-finite-sample-limitations)
- [Section 6: Scope and implications](#section-6-scope-and-implications)
- [Appendix: proof of the rate result](#appendix-proof-of-the-rate-result)
- [References](#references)

## Section 1: Model and finite-basis moments

### The conditional moment model

Let $D_t=(z_t,x_t,y_t)$ be i.i.d. observations, with
$z\in\mathbb R^{d_z}$, $x\in\mathbb R^{d_x}$, and

$$
y=g(\theta_*;x)+\varepsilon,
\qquad \theta_*\in\mathbb R^d,
\qquad \mathbb E[\varepsilon\mid z]=0.
\tag{1}
$$

The known structural map $g$ can be nonlinear in both $x$ and
$\theta$. Endogeneity is allowed: $\mathbb E[\varepsilon\mid x]$ need
not be zero. The conditional mean restriction in (1) is the assumption we
use; independence between the outcome and regressor conditional on the
instrument is not required. Nor do we need to specify or estimate a
first-stage regression for $x$ given $z$.

Choose a fixed vector of instrument functions
$\psi_{\mathcal K}(z)\in\mathbb R^{p_{\mathcal K}}$. Define the sample
moment, its sample Jacobian, and their expectations by

$$
\begin{aligned}
m_{\mathcal K}(D,\theta)
  &=\psi_{\mathcal K}(z)\{g(\theta;x)-y\},\\
M_{\mathcal K}(D,\theta)
  &=\psi_{\mathcal K}(z)\nabla_\theta g(\theta;x)^\top,\\
\bar m_{\mathcal K}(\theta)&=\mathbb E[m_{\mathcal K}(D,\theta)],
&\quad \bar M_{\mathcal K}(\theta)&=\mathbb E[M_{\mathcal K}(D,\theta)].
\end{aligned}
\tag{2}
$$

We use the residual $g-y$ throughout. Validity of the instruments gives
$\bar m_{\mathcal K}(\theta_{\ast})=0$. Assuming differentiation can pass
through expectation, $D_\theta\bar m_{\mathcal K}=\bar M_{\mathcal K}$.
The identity-weighted population criterion is

$$
Q_{\mathcal K}(\theta)=\frac12\|\bar m_{\mathcal K}(\theta)\|^2,
\qquad
\nabla Q_{\mathcal K}(\theta)
 =\bar M_{\mathcal K}(\theta)^\top\bar m_{\mathcal K}(\theta).
\tag{3}
$$

This criterion squares the **expected moment**. It is generally different
from $\mathbb E\lVert m_{\mathcal K}(D,\theta)\rVert^2/2$: minimizing the latter
need not recover $\theta_{\ast}$ under endogeneity. The distinction also
explains why multiplying a current sample Jacobian by the current sample
moment does not generally give an unbiased gradient of (3).

### A normalized Hermite basis

For the basis interpretation, suppose the selected functions are
orthonormal in $L^2(P_z)$:
$\mathbb E[\psi_{\mathcal K}(z)\psi_{\mathcal K}(z)^\top]=I$.
For $z\sim N(0,I_{d_z})$, one choice is the normalized tensor Hermite
system. With multi-index $\nu\in\mathbb N_0^{d_z}$, let

$$
\psi_\nu(z)=\prod_{j=1}^{d_z}
   \frac{\operatorname{He}_{\nu_j}(z_j)}{\sqrt{\nu_j!}},
\qquad
\mathcal I_{\mathcal K}
   =\{\nu:|\nu|_1\in\mathcal K\}.
\tag{4}
$$

Here $\mathcal K$ is a set of **total-degree blocks**. Selecting degree
two includes all squares and pairwise interactions. The first blocks are

$$
\psi_0(z)=1,\qquad
\psi_1(z)=z,\qquad
\psi_2(z)=\left(
 \frac{z_j^2-1}{\sqrt2},\ z_jz_\ell:\ j<\ell
\right)^\top.
\tag{5}
$$

Stack all features indexed by $\mathcal I_{\mathcal K}$ to obtain
$\psi_{\mathcal K}$. The moment dimension is

$$
p_{\mathcal K}
 =\sum_{k\in\mathcal K}\binom{d_z+k-1}{k}.
\tag{6}
$$

For example, with eight instruments, degree sets $\lbrace 1\rbrace$,
$\lbrace 0,1\rbrace$, $\lbrace 0,1,2\rbrace$, and $\lbrace 0,1,2,3\rbrace$ contain 8, 9, 45,
and 165 moments. Degree zero is the unconditional residual restriction;
it is present only when explicitly selected.

Orthonormality normalizes the moment coordinates. It does not make the
moment noise covariance the identity: at the truth that covariance is
$\mathbb E[\psi_{\mathcal K}\psi_{\mathcal K}^\top\varepsilon^2]$.
Thus identity weighting does not by itself imply efficient GMM. For
non-Gaussian instruments, polynomial features remain usable, but the
orthonormal projection formulas below require an orthonormalized system.
The streaming rate argument itself only needs the specified moment and
rank conditions.

## Section 2: A predictable Jacobian gives a one-observation update

Suppress $\mathcal K$ while holding the basis fixed. For ridge parameter
$\lambda>0$, define

$$
A_\lambda(S)=(S^\top S+\lambda I_d)^{-1}S^\top,
\qquad S\in\mathbb R^{p\times d}.
\tag{7}
$$

Given deterministic initial states $\theta_0\in\Theta$ and finite
$S_0$, use the recursion

$$
\begin{aligned}
\theta_{t+1}
 &=\Pi_\Theta\!\left[
     \theta_t-\alpha_{t+1}A_\lambda(S_t)m(D_{t+1},\theta_t)
   \right],\\
S_{t+1}
 &=(1-\beta_{t+1})S_t
       +\beta_{t+1}M(D_{t+1},\theta_t),\\
\bar\theta_T&=\frac1T\sum_{t=1}^T\theta_t.
\end{aligned}
\tag{8}
$$

The projection is Euclidean projection onto a deterministic compact convex
set $\Theta$ containing $\theta_{\ast}$. Averaging the parameter iterates
is standard in stochastic approximation
[Polyak and Juditsky (1992)][polyak]; we report a rate for both the last
iterate and this average below.

Each iteration reads one ordinary observation. Both sample quantities in
(8) are evaluated at the **old parameter** $\theta_t$, and the parameter
direction uses the **old Jacobian state** $S_t$. With
$\mathcal F_t=\sigma(D_1,\ldots,D_t)$, this ordering gives

$$
\mathbb E\!\left[
 A_\lambda(S_t)m(D_{t+1},\theta_t)\mid\mathcal F_t
\right]
 =A_\lambda(S_t)\bar m(\theta_t).
\tag{9}
$$

The moment noise and Jacobian noise from the new observation can be
correlated. Predictability of the gain is what permits (9); independent
samples for the two updates are unnecessary. In contrast, putting the
just-updated $S_{t+1}$ into the parameter direction generally invalidates
this conditional-mean calculation. If $S_0=0$, the first parameter step
is zero while the first observation initializes the Jacobian state.

Use two time scales:

$$
\alpha_t=a(t+t_0)^{-\eta},\qquad
\beta_t=b(t+t_0)^{-\kappa},\qquad
\frac12<\kappa<\eta\le1,
\tag{10}
$$

with $a,b>0$. Since $\alpha_t/\beta_t\to0$, the Jacobian adapts on
the faster time scale. Equation (9) is not an unbiased population gradient
or an exact population Gauss–Newton step: the state $S_t$ must learn the
changing target $\bar M(\theta_t)$. Its tracking error is part of the
convergence analysis.

The ridge controls numerical amplification because

$$
\|A_\lambda(S)\|_{\mathrm{op}}
 \le\max_{s\ge0}\frac{s}{s^2+\lambda}
 =\frac1{2\sqrt\lambda}.
\tag{11}
$$

It damps the gain without adding a penalty to the moment equation, so
$\theta_{\ast}$ remains a root. A fixed positive ridge does affect the
contraction rate. The persistent state uses $O(pd+d)$ memory, independent
of stream length; a direct dense implementation costs
$O(pd^2+d^3)$ per update, apart from evaluating the features and model.
Numerically, solve the system in (7) rather than forming an inverse.

This observation model differs from a conditional two-sample oracle, which
can draw independent regressors given the same instrument. Streaming IV
methods built around such an oracle, including the two-sample algorithm of
[Chen et al. (2024)][streaming-iv], provide useful context, but their
assumptions and sample budgets must be distinguished from (8).

## Section 3: Identification: which parameter directions do the moments see?

Define the conditional residual and its derivative at the truth:

$$
b_\theta(z)=\mathbb E[g(\theta;x)-y\mid z],
\qquad
a_*(z)=\mathbb E[\nabla_\theta g(\theta_*;x)\mid z].
\tag{12}
$$

Assume these functions are square integrable and conditional
differentiation is valid. Write $\Pi_{\mathcal K}$ for orthogonal
projection onto the span of the selected instrument functions. Then

$$
\|\bar m_{\mathcal K}(\theta)\|^2
 =\|\Pi_{\mathcal K}b_\theta\|_{L^2(P_z)}^2.
\tag{13}
$$

The selected moments detect only the projected conditional residual.
Consequently, **global identification on $\Theta$** is precisely the
separation condition

$$
\Pi_{\mathcal K}b_\theta\ne0
\quad\text{for every }\theta\in\Theta\setminus\{\theta_*\}.
\tag{14}
$$

Instrument validity guarantees a zero at the truth; it does not guarantee
(14). For local analysis, define

$$
\begin{aligned}
G_{\mathcal K}&=\bar M_{\mathcal K}(\theta_*),
&J_{\mathcal K}&=G_{\mathcal K}^\top G_{\mathcal K},\\
\mu_{\mathcal K}&=\lambda_{\min}(J_{\mathcal K})
 =\inf_{\|v\|=1}
   \|\Pi_{\mathcal K}[a_*^\top v]\|_{L^2(P_z)}^2.
\end{aligned}
\tag{15}
$$

The number $\mu_{\mathcal K}$ measures signal in the least visible
parameter direction. A large number of moments is useful only if they
collectively capture every such direction.

### Result 1: regular local identification

Suppose $\bar M_{\mathcal K}$ is
$L_{\mathcal K}$-Lipschitz in Frobenius norm on a neighborhood of
$\theta_{\ast}$. If $\mu_{\mathcal K}>0$, the selected moments locally
identify $\theta_{\ast}$. More quantitatively, for any displacement $e$
whose connecting segment remains in that neighborhood,

$$
\|\bar m_{\mathcal K}(\theta_*+e)\|
 \ge\sqrt{\mu_{\mathcal K}}\|e\|
       -\frac{L_{\mathcal K}}2\|e\|^2.
\tag{16}
$$

**Proof.** The integral Taylor formula gives
$\bar m_{\mathcal K}(\theta_{\ast}+e)=G_{\mathcal K}e+R(e)$, with
$\lVert R(e)\rVert\le L_{\mathcal K}\lVert e\rVert^2/2$. Since
$\lVert G_{\mathcal K}e\rVert\ge\sqrt{\mu_{\mathcal K}}\lVert e\rVert$, the triangle
inequality gives (16). Its right-hand side is positive for small nonzero
$e$. If the population moment is affine in $\theta$, the remainder
vanishes, and full column rank gives global identification on the parameter
set. $\square$

This is the familiar full-column-rank sufficient condition for regular
local identification; see [Chen et al. (2014, Section 2.1)][identification].
The dimension requirement $p_{\mathcal K}\ge d$ is necessary for this
rank condition, but is not sufficient. Conversely, rank deficiency does
not rule out identification through higher-order terms. Such singular
identification is outside the regular rate analysis in Section 4.

### A basis can miss signal even in one dimension

Consider a scalar regressor with
$x=\gamma(z^2-1)+u$, $z\sim N(0,1)$, and structural map
$g(\theta;x)=\sin(\theta x)$ at $\theta_{\ast}=0$. Let centered jointly
Gaussian $(u,\varepsilon)$ be independent of $z$, with
$\operatorname{Cov}(u,\varepsilon)\ne0$. The instrument is valid while
the regressor is endogenous. Here

$$
a_*(z)=\gamma(z^2-1),\qquad
G_{\{1\}}=0,\qquad
G_{\{2\}}=\sqrt2\gamma,
\qquad \mu_{\{2\}}=2\gamma^2.
\tag{17}
$$

In fact, the degree-one moment is zero for every parameter: the joint law
is invariant under $z\mapsto-z$, while that instrument is odd. Degree
two supplies regular local identification whenever $\gamma\ne0$.
This is a failure of the chosen moment set, despite having
$p_{\lbrace 1\rbrace}=d=1$.

### Local rank does not establish global uniqueness

For a separate example, take $x=z+u$ with independent standard Gaussian
$z,u$, and let
$g(\theta;x)=\theta x+\theta^2(x^2-2)$. Under a valid structural error,

$$
b_\theta(z)
 = (\theta-\theta_*)z
   +(\theta^2-\theta_*^2)(z^2-1).
\tag{18}
$$

If $\theta_{\ast}\ne0$, the degree-two moment has a nonzero derivative at the
truth, yet it also vanishes at $-\theta_{\ast}$. A parameter set containing
both values is therefore not globally identified by degree two alone.
Adding degree one separates the two roots. A local rank calculation and a
global separation argument answer different questions.

### What enlarging the basis can guarantee

For orthonormal degree blocks, write
$C_k=\mathbb E[\psi_k(z)a_{\ast}(z)^\top]$. Then

$$
J_{\mathcal K}=\sum_{k\in\mathcal K}C_k^\top C_k.
\tag{19}
$$

Nested basis expansions increase $J_{\mathcal K}$ in positive
semidefinite order. If the full orthonormal system is complete and
$J_\infty=\mathbb E[a_{\ast}a_{\ast}^\top]\succ0$, orthogonal decomposition gives

$$
\mu_{\mathcal K}
 \ge \lambda_{\min}(J_\infty)
       -\mathbb E\|a_*-\Pi_{\mathcal K}a_*\|^2.
\tag{20}
$$

Indeed, $J_\infty-J_{\mathcal K}$ is the second-moment matrix of the
omitted sensitivity, whose operator norm is bounded by its squared
$L^2$ norm. Thus some finite initial segment gives regular local
identification. Completeness is a way to obtain a sufficiently rich finite
basis; it is not a requirement that every valid application use infinitely
many moments.

## Section 4: Convergence rates: identification, curvature, and tracking

Identification concerns the zeros of the population moments. Convergence
also depends on the geometry of the update and the region visited by the
iterates. The following conditions make that distinction explicit.

### Assumptions and the admissible region

Fix the basis. Vector norms below are Euclidean, and matrix norms are
Frobenius unless marked $\mathrm{op}$. In addition to the i.i.d. model
and differentiability in Section 1, assume:

1. **A fixed compact region.** $\Theta$ is deterministic, compact, and
   convex, contains $\theta_{\ast}$, and contains the deterministic
   initialization $\theta_0$. The initial Jacobian state $S_0$ is
   finite. The algorithm uses the projection in (8).
2. **Population smoothness.** On $\Theta$,
   $D_\theta\bar m=\bar M$ and
   $\lVert \bar M(\theta)-\bar M(\theta^\prime)\rVert_F\le L\lVert \theta-\theta^\prime\rVert$.
3. **Uniform fourth moments.** The constants

   $$
   \sigma_m^4=\sup_{\theta\in\Theta}\mathbb E\|m(D,\theta)\|^4,
   \qquad
   \sigma_M^4=\sup_{\theta\in\Theta}
      \mathbb E\|M(D,\theta)-\bar M(\theta)\|_F^4
   \tag{21}
   $$

   are finite.
4. **Regular identification and controlled curvature.** With
   $G=\bar M(\theta_{\ast})$ and $\mu=\lambda_{\min}(G^\top G)>0$, set

   $$
   h=\frac{\mu}{\mu+\lambda},\qquad
   \ell_\lambda=\frac{5}{4\lambda},\qquad
   c=L\!\left(\ell_\lambda\|G\|_{\mathrm{op}}
                +\frac1{4\sqrt\lambda}\right).
   \tag{22}
   $$

   Require $\Theta\subseteq B(\theta_{\ast},r)$ with
   $rc\le h/4$. If $L=0$, this radius restriction is vacuous.
5. **Separated step sizes.** Use (10), with $t_0\ge1$ sufficiently large
   that $0<\beta_t\le1$ and $h\alpha_t\le1$ for every update.

The local region is a substantive assumption. A large projection ball
centered at zero need not meet it, and the theorem does not provide an
algorithm for locating a neighborhood of the unknown truth. For an
unprojected implementation, the same conclusions require a separate proof
that all iterates stay in the specified region almost surely. A
population rank condition alone does not supply this stability property.

### Result 2: finite-time mean-square convergence

Under these assumptions, define

$$
\rho=\min\{\eta,\ 2\kappa,\ 4(\eta-\kappa)\}>0.
\tag{23}
$$

If $\eta<1$, then

$$
\mathbb E\|\theta_T-\theta_*\|^2=O(T^{-\rho}),
\qquad
\mathbb E\|\bar\theta_T-\theta_*\|^2=O(T^{-\rho}).
\tag{24}
$$

If $\eta=1$, put $\gamma=ah$. For both the last iterate and its
average, the mean-square error is bounded by

$$
\begin{cases}
O(T^{-\rho}),&\gamma>\rho,\\
O(T^{-\rho}\log T),&\gamma=\rho,\\
O(T^{-\gamma}),&\gamma<\rho.
\end{cases}
\tag{25}
$$

All implied constants can depend on the fixed basis, the parameter region,
noise moments, ridge, initialization, and step-size coefficients; they do
not depend on $T$. If $\eta+\rho>1$, the last iterate also converges to
$\theta_{\ast}$ almost surely. The proof is given in the appendix.

A useful specialization is

$$
\alpha_t=\frac{a}{t+t_0},\qquad
\beta_t=\frac{b}{(t+t_0)^{2/3}},\qquad
ah>1.
\tag{26}
$$

Then there is a finite constant $C_{\mathcal K}$ such that

$$
\mathbb E\|\theta_T-\theta_*\|^2
 \le\frac{C_{\mathcal K}}{T+t_0},
\qquad
\mathbb E\|\bar\theta_T-\theta_*\|^2
 \le\frac{4C_{\mathcal K}}{T}.
\tag{27}
$$

Thus the root-mean-square parameter error is $O(T^{-1/2})$. The
statement is about expectations over repeated streams. It is not a
pointwise guarantee for every trajectory, nor a claim of asymptotic
normality or efficient covariance. In this proof, averaging preserves the
rate order; it is not used to establish an efficiency improvement.

### Why the exponent has three terms

Let $\delta_t=S_t-\bar M(\theta_t)$. The analysis bounds its fourth
moment as

$$
\mathbb E\|\delta_t\|_F^4
 =O\!\left((t+t_0)^{-2\kappa}
            +(t+t_0)^{-4(\eta-\kappa)}\right).
\tag{28}
$$

The first term is Jacobian-estimation noise; the second is lag from
tracking a Jacobian that moves with the parameter. The direct parameter
noise contributes the $\eta$ term in (23). Controlling the chance of a
large tracking error combines these three contributions into (24)–(25).
These are sufficient upper bounds, and need not equal the sharp exponent
observed in a particular model.

When the structural model is linear in its parameters,
$g(\theta;x)=q(x)^\top\theta$, the population Jacobian is constant.
Then $L=0$, the moving-target term disappears, and for $\eta<1$ the
bound improves to $O(T^{-\eta})$ for both estimators. This conclusion
holds on the chosen compact region without the local radius restriction.
A function quadratic or otherwise nonlinear in $x$ can still belong
to this simpler case; nonlinearity in the regressors and nonlinearity in
the estimated parameters must be distinguished.

For genuinely nonlinear parameters, $\eta=0.7,\kappa=0.6$ gives the
conservative exponent $\rho=0.4$, while
$\eta=0.9,\kappa=0.6$ gives $\rho=0.9$. More generally, for
$2/3<\eta<1$, taking $1/2<\kappa\le3\eta/4$ yields $\rho=\eta$.
The harmonic setting (26) yields $1/T$; any
$1/2<\kappa\le3/4$ gives the same order if $ah>1$.

### How the basis affects the rate guarantee

The identification strength from Result 1 reappears in the contraction
factor $h=\mu/(\mu+\lambda)$. For a fixed harmonic coefficient $a$,
the sufficient condition for (27) is

$$
ah>1
\quad\Longleftrightarrow\quad
a>1\ \text{ and }\ \mu>\frac{\lambda}{a-1}.
\tag{29}
$$

A basis can therefore identify the parameter while having too little
signal to satisfy this particular $1/T$ guarantee with the chosen
coefficient. Adjusting $a$ to restore $ah>1$ changes the step-size
constants and can lengthen the transient; it is not a cost-free remedy.

Adding basis blocks can increase $\mu$, but can also increase the
fourth moments in (21) and the curvature constant. If an added block has
$C_k=0$, it supplies no additional first-order signal. At the population
level,

$$
A_\lambda\!\left(\begin{bmatrix}G\\0\end{bmatrix}\right)
 =\begin{bmatrix}A_\lambda(G)&0\end{bmatrix},
\tag{30}
$$

whereas its estimated rows fluctuate at finite sample sizes. The theory
therefore supports checking signal, curvature, and noise together. It
does not imply that a larger basis always improves estimation.

## Section 5: Experiments: identification, rates, and finite-sample limitations

The experiments follow the two theoretical questions in that order. First,
a scalar model lets us verify identification and every condition of the
local rate result analytically. We then test the effect of basis expansion
in a multivariate model, and use a nonlinear exponential model to examine
initialization and the limits of local conclusions.

All runs use the one-observation recursion (8), with the previous Jacobian
state, parameter projection, and the average of all updated iterates. The
compiled implementation was checked against the existing SSGMM code on
identical streams. No configuration uses direction clipping, exponential
clipping, restarts, or early stopping. The full settings, fixed parameter
vectors, seeds, and code version are recorded in the
[experiment protocol](../files/sieve-sgmm/protocol.json).
The [terminal statistics](../files/sieve-sgmm/terminal-statistics.csv)
and [budget comparisons](../files/sieve-sgmm/budget-statistics.csv)
include all configurations, both estimators, and the uncertainty diagnostics.

### Statistical protocol

The scalar groups use 512 independent repeated streams, each of length
$T=2^{18}$. Multivariate groups use 64 repetitions, first at $2^{18}$
and then through the prespecified $2^{19}$ and $2^{20}$ budgets.
Configurations share Gaussian draws within a repetition; repetitions are
independent. The extended streams reproduce the initial $2^{18}$
parameter and average checkpoints exactly. The master seed is 20261023;
initialization, pilot estimation, and evaluation use separate random streams.

The primary statistic is the arithmetic mean of **squared parameter
errors** across repetitions. Monte Carlo standard errors (MCSEs) are
computed from those squared errors. Shading in the figures is the
pointwise normal approximation $\widehat{\mathrm{MSE}}\pm1.96\,\mathrm{MCSE}$,
not a simultaneous confidence band. We also report percentile bootstrap
intervals using 10,000 resamples of whole repeated trajectories
[Efron (1979)][efron]. Resampling individual, dependent time checkpoints
would give a different and inappropriate uncertainty calculation.
Descriptive log–log slopes use the fixed terminal window $T/4\le t\le T$.
They do not establish an asymptotic rate.

An independent fixed test set of $2^{15}$ observations evaluates
structural risk,
$\mathbb E[(g(\hat\theta;x)-g(\theta_{\ast};x))^2]$.
Outcome prediction MSE is auxiliary: under endogeneity its residual cross
term need not vanish, so it cannot replace parameter recovery or
structural risk. Repeated-stream intervals do not include pilot uncertainty
or finite-test-set integration error. All repetitions were retained; there
were no unrecovered numerical failures.

### A calibrated nonlinear model

Let $z,u,\nu$ be independent standard Gaussian variables and set

$$
x=\gamma z+u,\qquad
\varepsilon=0.5u+0.5(u^2-1)+\nu,\qquad
y=g(0;x)+\varepsilon,
\qquad g(\theta;x)=(\theta+0.01\theta^2)x^2.
$$

This model is nonlinear in its parameter. It has
$\mathbb E[\varepsilon\mid z]=0$,
$\operatorname{Cov}(x,\varepsilon)=0.5$, and
$\mathbb E[x^2\varepsilon]=1$; even the structural feature is endogenous.
Use $\Theta=[-1,1]$, $\theta_0=0.5$, $S_0=0$,
$\lambda=1$, $t_0=100$,
$\alpha_t=a/(t+100)$, and $\beta_t=(t+100)^{-2/3}$.

The degree-one moment is identically zero. For degree two,
$\bar m_2(\theta)=\sqrt2\gamma^2(\theta+0.01\theta^2)$ and
$\mu_2=2\gamma^4$. The moment has a unique zero on the parameter region.
Its Lipschitz constant is $L=0.02\sqrt2\gamma^2$. For
$\gamma=1$ and $0.5$, the constants in (22) satisfy the radius
condition with $r=1$; all required fourth moments are finite. Thus the
identifying runs provide a setting in which the theorem's region and
moment assumptions can actually be checked.

| Moment | $\gamma$ | $a$ | Last-iterate MSE | Averaged-iterate MSE | Last-iterate slope |
|:--|--:|--:|--:|--:|--:|
| Degree 1 | 1 | 3 | 0.2712 | 0.2711 | 0.001 |
| Degree 2 | 1 | 3 | 4.692e-06 | 5.016e-06 | -1.054 |
| Degree 2 | 0.5 | 3 | 0.003134 | 0.006832 | -0.671 |
| Degree 2 | 0.5 | 12 | 5.955e-05 | 0.0001083 | -1.085 |
| Degree 2 | 0.5 | 9 | 5.769e-05 | 0.0001579 | -1.110 |

The unidentifying degree-one moment does not recover the parameter.
With degree two and $\gamma=1$, $h=2/3$ and $ah=2>1$;
the observed error is compatible with the $1/T$ setting. Its terminal
MSE has a repeat-bootstrap 95% interval of
$[4.16,5.25]\times10^{-6}$.

Reducing $\gamma$ to $0.5$ gives $h=1/9$. Keeping $a=3$ produces
much slower recovery; raising it to 12 restores the sufficient
$ah>1$ condition and improves the finite-sample error. The observed
slope near $-0.67$ in the fixed-coefficient weak-signal case is faster
than the theorem's conservative $-1/3$ upper-bound exponent. This is
allowed by the bound. The boundary choice $a=9$ is also retained;
these finite curves cannot establish that a logarithmic factor must
appear when $ah=1$.

[![Identification and rate calibration: last, averaged, scaled errors, and terminal distributions][calibration-rates-png]][calibration-rates-svg]

*Scalar calibration, 512 repetitions. Both estimators and all five
configurations are shown. The dotted line is a scale reference; the
terminal distributions retain every repeated stream.*

### Tracking need not determine the observed rate

Using the same scalar DGP, we compare exponent pairs
$(\eta,\kappa)=(0.7,0.6),(0.9,0.6),(1,2/3)$, both with curvature
0.01 and with the parameter-linear map $g(\theta;x)=\theta x^2$.
An additional diagnostic uses the exact population Jacobian in the gain;
it has information unavailable to the estimator.

The estimated- and population-Jacobian last-iterate curves are close.
For example, the estimated-Jacobian slopes in the nonlinear model are
approximately $-0.685,-0.894,-0.998$, respectively. Here tracking is
not the dominant finite-sample error, and the nonlinear model has small
curvature. Consequently, these experiments do not establish sharpness
of the moving-target term in (28). The
[tracking fourth-moment plot](../images/sieve-sgmm/tracking-scales.svg)
shows the observed errors alongside the theoretical time scales, with
arbitrary vertical constants. Faster empirical behavior does not improve
the general theorem.

### Multivariate basis expansion: signal and noise have different effects

Use $d_x=4,d_z=8$ and the 14-parameter symmetric quadratic structure
$g(\theta;x)=b^\top x+x^\top A x$. Let
$x=\phi(\Gamma^\top z)+c+v$, with independent
$z\sim N(0,I_8)$, $c,v\sim N(0,0.5I_4)$,
$e\sim N(0,1)$, and structural error
$\varepsilon=0.5\mathbf1^\top c+e$.
The fixed full-column-rank $\Gamma$ and true $b$ are generated with
parameter seed 10, with $\Gamma_{ij}\sim N(0,1)$ and
$b_{\ast}\sim N(0,4I_4)$; $A_{\ast}=I_4$. Their actual values are in the protocol.
Initialize $b$ from $N(0,0.01I_4)$ and $A=0$, and project onto
the radius-20 ball about zero, which contains the truth. Although
quadratic in $x$, this model is linear in $\theta$, so $L=0$.

Compare complete degree sets $\lbrace 1\rbrace,\lbrace 0,1\rbrace,\lbrace 0,1,2\rbrace,\lbrace 0,1,2,3\rbrace$,
with 8, 9, 45, and 165 moments. Both a power schedule
$(a,\eta,\kappa)=(0.1,0.9,0.6)$ and a harmonic diagnostic are retained,
with $\lambda=0.01$ and offset at least 100. The harmonic diagnostic
uses $\eta=1,\kappa=2/3$. In that diagnostic,
a shared coefficient is set before training to
$a=2/\min(h_{012},h_{0123})$, using exact signal for the linear first
stage and independent pilot estimates for ReLU. The latter does not
certify the exact population threshold.

For $\phi(s)=s$, the exact ranks are **4, 5, 14, 14**. The last two
sets have the same $\mu=0.12547$, and the degree-three Jacobian block
is exactly zero. Under the harmonic schedule, their terminal last-iterate
MSEs at $2^{20}$ are respectively **2.256e-5** and **5.494e-5**.
H012's terminal slope is approximately $-1.03$. H0123 is still in a
steeper transient; its extra zero-signal rows do not improve recovery at
this budget. The small-coefficient power runs and the full-history averages
also retain substantial transient effects. They cannot be treated as
measurements of a sharp asymptotic exponent.

[![All basis sets and both schedules in the quadratic-linear model][linear-basis-rates-png]][linear-basis-rates-svg]

*Linear first stage, 64 repetitions through $2^{20}$ observations.
The first two moment sets are rank deficient. The last two have identical
population identification strength; degree three adds sampling fluctuations.*

For $\phi(s)=\max(s,0)$, higher degrees can capture additional
conditional sensitivity. Two independent Rao–Blackwell pilots, each with
$2^{20}$ instruments, estimate $\mu_{012}\approx0.000652$ and
$\mu_{0123}\approx0.0129$. The estimated degree-three block has
Frobenius norm about 7.275. Pilot batch means and their Monte Carlo errors
are retained; a numerically nonzero singular value is not a proof of
population full rank.

At $2^{20}$, the harmonic last-iterate MSE is **0.3039** for H012 and
**0.001597** for H0123. Their repeat-bootstrap 95% intervals are
approximately $[0.122,0.544]$ and $[0.00122,0.00203]$.
H012's mean is sensitive to large-error repetitions: its largest run
contributes about 28% of terminal MSE, down from about 44% at $2^{18}$.
Reporting only a median would hide this feature.

[![All basis sets and both schedules in the quadratic-ReLU model][relu-basis-rates-png]][relu-basis-rates-svg]

*ReLU first stage, 64 repetitions. Both schedules and estimators are
retained, including slow and high-error finite-sample trajectories. The
signal calculations use independent pilots and carry estimation uncertainty.*

Together, the two first stages support the signal–noise distinction:
degree three adds no population signal in the linear case, whereas it
adds sensitivity and improves recovery in this ReLU setting. They do not
support a universal preference for a larger basis.

### A nonlinear-parameter stress test: initialization remains consequential

Finally, take $g(\theta;x)=\exp(\theta^\top x)$ and
$x=\tanh(\Gamma^\top z)+c+v$, with the same dimensions and Gaussian
first-stage disturbances, $\Gamma_{ij}\sim N(0,0.1^2)$, and
$\varepsilon=0.25\mathbf1^\top c+0.1e$.
A separate pilot scales the fixed truth to target index variance 0.25.
Use $a=0.1,\eta=0.9,\kappa=0.6,\lambda=0.01,t_0=100$.
Near-zero initial values follow $N(0,0.01I_4)$; near-truth values follow
$N(\theta_{\ast},0.025^2I_4)$. Compare these two initializations,
both in the radius-2 ball about zero; an idealized local diagnostic uses
a radius-0.1 ball about the truth. That last region uses simulated truth
and is not an implementable initialization strategy.

For H01 at $2^{20}$, last-iterate MSE is **0.1119** from near zero and
**0.0002589** from near the truth. The local-projection version gives
**0.0002587**. H0123 near the truth gives **0.0007958**, so expansion is
not uniformly beneficial here either. The local and near-truth runs are
close because the small projection ball rarely activates.

[![Exponential nonlinear model: all basis sets, three initialization/region choices, last and averaged iterates][exp-initialization-png]][exp-initialization-svg]

*Exponential model, 64 repetitions. These are initialization and region
diagnostics. The curvature-radius inequality has not been verified for
these configurations, so the figure is not presented as a test of a global
nonlinear convergence theorem.*

This stress test illustrates why identification and local convergence
conditions should not be collapsed into a global recovery claim.
Differences in these finite trajectories alone do not establish their
underlying global geometry. The calibrated scalar experiment tests the
stated local conditions; this more complex example records what happens
when those conditions have not been established.

## Section 6: Scope and implications

The two results give a sequence of checks for a streaming IV problem.
First ask whether the selected moments detect every local parameter
direction, using $\mu_{\mathcal K}$. If global recovery is required,
check separation on the whole parameter set as well. Then establish that
the algorithm remains in a region where its mean direction contracts, and
choose step sizes that account for Jacobian tracking.

This distinction matters even with globally identifying moments. For
example, the deterministic scalar moment $\bar m(\theta)=\theta^3-1$
has a unique root at one. Nevertheless, starting (8) at
$\theta_0=0,S_0=0$ leaves both states unchanged, because the sample
Jacobian there is zero. Global identification alone cannot rule out
stationary points of the iteration.

The analysis concerns a correctly specified finite-dimensional structural
model, i.i.d. data, and a fixed basis chosen before the stream. Once that
basis identifies $\theta_{\ast}$, finite truncation creates no automatic
approximation bias in the parameter. Growing or adaptively selected
bases, structural misspecification, dependent observations, and singular
identification require additional analysis. None of the bounds above
establishes those extensions or global nonlinear convergence from an
arbitrary initialization.

## Appendix: proof of the rate result

The proof below makes the tracking and stability requirements explicit.
All constants are finite and independent of time; their values may depend
on the fixed basis. Write $n_t=t+t_0$ and
$u_t=\mathbb E\lVert \theta_t-\theta_{\ast}\rVert^2$.

### A. Gain regularity and local contraction

Besides (11), the damped gain is globally Lipschitz:

$$
\|A_\lambda(S)-A_\lambda(T)\|_{\mathrm{op}}
 \le\ell_\lambda\|S-T\|_F,
\qquad \ell_\lambda=\frac5{4\lambda}.
\tag{31}
$$

To verify this, set $P=(S^\top S+\lambda I)^{-1}$. Differentiation gives

$$
DA_\lambda(S)[E]
 =PE^\top(I-SA_\lambda(S))-A_\lambda(S)EA_\lambda(S).
\tag{32}
$$

Use $\lVert P\rVert_{\mathrm{op}}\le1/\lambda$,
$\lVert I-SA_\lambda(S)\rVert_{\mathrm{op}}\le1$, and (11), then integrate
along the line segment between the two matrices.

Let $e=\theta-\theta_{\ast}$ and $H=A_\lambda(G)G$. Its eigenvalues are
$s^2/(s^2+\lambda)$, so $H\succeq hI$. Taylor expansion and (31)
yield

$$
\|A_\lambda(\bar M(\theta))\bar m(\theta)-He\|
 \le c\|e\|^2.
\tag{33}
$$

The radius condition $rc\le h/4$ therefore gives the drift inequality

$$
e^\top A_\lambda(\bar M(\theta))\bar m(\theta)
 \ge\frac{3h}{4}\|e\|^2
\qquad\text{on }\Theta.
\tag{34}
$$

### B. Tracking the moving Jacobian

Define
$\xi_i=M(D_i,\theta_{i-1})-\bar M(\theta_{i-1})$. These are
martingale differences with conditional fourth moments bounded by
$\sigma_M^4$. The state error satisfies

$$
\delta_t=(1-\beta_t)\delta_{t-1}+\beta_t\xi_t
       +\bar M(\theta_{t-1})-\bar M(\theta_t).
\tag{35}
$$

Let
$\pi_{i,t}=\prod_{j=i+1}^t(1-\beta_j)$ and
$\omega_{i,t}=\beta_i\pi_{i,t}$, with an empty product equal to one.
Unrolling (35) separates initialization, sample noise, and target motion:

$$
\delta_t=\pi_{0,t}\delta_0
 +\sum_{i=1}^t\omega_{i,t}\xi_i
 +\sum_{i=1}^t\pi_{i,t}
    \{\bar M(\theta_{i-1})-\bar M(\theta_i)\}.
\tag{36}
$$

Conditional expansion of the fourth power gives
$\mathbb E\lVert \sum_i\omega_{i,t}\xi_i\rVert_F^4 \le 8\sigma_M^4(\sum_i\omega_{i,t}^2)^2$.
For example, write $Q_{i-1}=\sum_{j<i}w_j^2$ for deterministic
weights. Martingale orthogonality bounds the partial sum's second moment
by $\sigma_M^2 Q_{i-1}$; the increment in its fourth moment is at
most $\sigma_M^4(6w_i^2Q_{i-1}+4\lvert w_i\rvert^3\sqrt{Q_{i-1}}+w_i^4)$.
Summing these terms bounds them by eight times
$\sigma_M^4(\sum_iw_i^2)^2$.
For the last term, projection is nonexpansive and fixes
$\theta_{i-1}$, so (11) implies

$$
\bigl(\mathbb E\|\theta_i-\theta_{i-1}\|^4\bigr)^{1/4}
 \le\frac{\alpha_i\sigma_m}{2\sqrt\lambda}.
\tag{37}
$$

Combining smoothness with Minkowski's inequality yields

$$
\mathbb E\|\delta_t\|_F^4
 \le C\!\left[
    \pi_{0,t}^4\|\delta_0\|_F^4
    +\sigma_M^4 W_t^2
    +\left(\frac{L\sigma_m}{2\sqrt\lambda}\right)^4 U_t^4
  \right]=:\mathcal R_t,
\tag{38}
$$

where $W_t=\sum_i\omega_{i,t}^2$ and
$U_t=\sum_i\alpha_i\pi_{i,t}$. The scalar recursions

$$
W_t=(1-\beta_t)^2W_{t-1}+\beta_t^2,
\qquad
U_t=(1-\beta_t)U_{t-1}+\alpha_t
\tag{39}
$$

give $W_t=O(\beta_t)$ and
$U_t=O(\alpha_t/\beta_t)$. One can verify these bounds by induction:
the comparison sequences change by a relative $O(1/t)$, while
$1/t=o(\beta_t)$. Initialization decays faster than any power because
$\pi_{0,t}\le\exp(-\sum_{j=1}^t\beta_j)$. This proves (28). The
argument never assumes independence between sample noise and target
motion. If $L=0$, the last term of (38) is exactly zero.

### C. From tracking to parameter error

Set

$$
B=\|G\|_{\mathrm{op}}+Lr,\qquad
q=\frac{h}{4\ell_\lambda B},\qquad
\mathcal B_t=\{\|\delta_t\|_F>q\}.
\tag{40}
$$

Smoothness gives $\lVert \bar m(\theta)\rVert\le B\lVert \theta-\theta_{\ast}\rVert$.
Equations (31) and (34) then imply
$e_t^\top A_\lambda(S_t)\bar m(\theta_t)\ge h\lVert e_t\rVert^2/2$ on
$\mathcal B_t^c$. On $\mathcal B_t$, boundedness of the region and
gain gives the uniform replacement

$$
2e_t^\top A_\lambda(S_t)\bar m(\theta_t)
 \ge h\|e_t\|^2-w\mathbf1_{\mathcal B_t},
\qquad w=r^2(h+B/\sqrt\lambda).
\tag{41}
$$

Expand the squared projected parameter update and take conditional
expectations using (9). With $v=\sigma_m^2/(4\lambda)$,

$$
\mathbb E[\|e_{t+1}\|^2\mid\mathcal F_t]
 \le(1-h\alpha_{t+1})\|e_t\|^2
     +v\alpha_{t+1}^2
     +w\alpha_{t+1}\mathbf1_{\mathcal B_t}.
\tag{42}
$$

Markov's inequality bounds
$\Pr(\mathcal B_t)\le\mathcal R_t/q^4$. Consequently,

$$
u_{t+1}\le(1-h\alpha_{t+1})u_t
       +v\alpha_{t+1}^2
       +\frac{w}{q^4}\alpha_{t+1}\mathcal R_t.
\tag{43}
$$

This is also an explicit finite-time bound: with
$P_{i,T}=\prod_{j=i+1}^T(1-h\alpha_j)$,

$$
u_T\le P_{0,T}u_0+
 \sum_{i=1}^T P_{i,T}
   \left[v\alpha_i^2+
         \frac{w}{q^4}\alpha_i\mathcal R_{i-1}\right].
\tag{44}
$$

Its constants expose the dependence on identification strength, curvature,
moment noise, and initialization. In particular, the result is not uniform
over a sequence of increasingly weakly identified bases.

### D. Solving the recursion and averaging

The forcing term in (43) is $O(n_{t+1}^{-\eta-\rho})$.
For $\eta<1$, comparison with a sufficiently large multiple of
$n_t^{-\rho}$ proves (24), since its relative decrease is
$O(1/t)=o(\alpha_t)$. For $\eta=1$,

$$
P_{i,T}\le C\left(\frac{i+t_0}{T+t_0}\right)^{ah}.
\tag{45}
$$

Substitution into (44) yields the three cases in (25). Under (26),
$\mathcal R_t=O(n_t^{-4/3})$, so the constant

$$
D=\sup_{t\ge0} n_{t+1}^2
 \left[v\alpha_{t+1}^2+
       \frac{w}{q^4}\alpha_{t+1}\mathcal R_t\right]
\tag{46}
$$

is finite. Induction in (43) with
$C_{\mathcal K}=\max\lbrace t_0u_0,D/(ah-1)\rbrace$ proves the first inequality
in (27). Minkowski's inequality gives

$$
\bigl(\mathbb E\|\bar\theta_T-\theta_*\|^2\bigr)^{1/2}
 \le\frac1T\sum_{t=1}^T\sqrt{u_t}.
\tag{47}
$$

This preserves the powers and boundary logarithm in (24)–(25); when
$u_t\le C_{\mathcal K}/(t+t_0)$, the right-hand side is at most
$2\sqrt{C_{\mathcal K}/T}$, proving the second inequality in (27).

Finally, if $\eta+\rho>1$, (38) implies
$\sum_t\alpha_{t+1}\Pr(\mathcal B_t)<\infty$ and hence
$\sum_t\alpha_{t+1}\mathbf1_ {\mathcal B_t}<\infty$ almost surely.
Apply the almost-supermartingale theorem of
[Robbins and Siegmund (1971)][robbins-siegmund] to (42). It gives
convergence of $\lVert e_t\rVert^2$ and finiteness of
$\sum_t\alpha_{t+1}\lVert e_t\rVert^2$. Since
$\sum_t\alpha_t=\infty$, that limit is zero. $\square$

## References

1. Hansen, L. P. (1982). [Large sample properties of generalized method of moments estimators][hansen]. *Econometrica*, **50**(4), 1029–1054.
2. Chen, X., Chernozhukov, V., Lee, S., and Newey, W. K. (2014). [Local identification of nonparametric and semiparametric models][identification]. *Econometrica*, **82**(2), 785–809. DOI: 10.3982/ECTA9988.
3. Chen, X., Lee, S., Liao, Y., Seo, M. H., Shin, Y., and Song, M. (2025). [SGMM: Stochastic approximation to generalized method of moments][sgmm]. *Journal of Financial Econometrics*, **23**(1), nbad027. DOI: 10.1093/jjfinec/nbad027. The nonlinear construction is discussed in Section 7.
4. Chen, X., Roy, A., Hu, Y., and Balasubramanian, K. (2024). [Stochastic optimization algorithms for instrumental variable regression with streaming data][streaming-iv]. *Advances in Neural Information Processing Systems*, **37**, 26510–26542.
5. Borkar, V. S. (1997). [Stochastic approximation with two time scales][borkar]. *Systems & Control Letters*, **29**(5), 291–294. DOI: 10.1016/S0167-6911(97)90015-3.
6. Polyak, B. T., and Juditsky, A. B. (1992). [Acceleration of stochastic approximation by averaging][polyak]. *SIAM Journal on Control and Optimization*, **30**(4), 838–855. DOI: 10.1137/0330046.
7. Robbins, H., and Siegmund, D. (1971). [A convergence theorem for non negative almost supermartingales and some applications][robbins-siegmund]. In J. S. Rustagi (Ed.), *Optimizing Methods in Statistics*, pp. 233–257. Academic Press. DOI: 10.1016/B978-0-12-604550-5.50015-8.

8. Efron, B. (1979). [Bootstrap methods: Another look at the jackknife][efron]. *The Annals of Statistics*, **7**(1), 1–26. DOI: 10.1214/aos/1176344552.

[hansen]: https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/
[identification]: https://doi.org/10.3982/ECTA9988
[sgmm]: https://doi.org/10.1093/jjfinec/nbad027
[streaming-iv]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ea07a4acbf7e38913062fd69a70805f-Abstract-Conference.html
[borkar]: https://doi.org/10.1016/S0167-6911(97)90015-3
[polyak]: https://doi.org/10.1137/0330046
[robbins-siegmund]: https://doi.org/10.1016/B978-0-12-604550-5.50015-8

[efron]: https://doi.org/10.1214/aos/1176344552

[calibration-rates-png]: ../images/sieve-sgmm/calibration-rates.png
[calibration-rates-svg]: ../images/sieve-sgmm/calibration-rates.svg
[linear-basis-rates-png]: ../images/sieve-sgmm/linear-basis-rates.png
[linear-basis-rates-svg]: ../images/sieve-sgmm/linear-basis-rates.svg
[relu-basis-rates-png]: ../images/sieve-sgmm/relu-basis-rates.png
[relu-basis-rates-svg]: ../images/sieve-sgmm/relu-basis-rates.svg
[exp-initialization-png]: ../images/sieve-sgmm/exp-initialization.png
[exp-initialization-svg]: ../images/sieve-sgmm/exp-initialization.svg
