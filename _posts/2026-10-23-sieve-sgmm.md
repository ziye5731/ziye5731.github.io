---
title: "Sieve SGMM: Online IV Estimation through Basis Moments"
date: 2026-10-23
permalink: /posts/2026/10/sieve-sgmm/
excerpt: "A one-observation streaming method for nonlinear instrumental variable regression: how basis moments, a past Jacobian, and two time scales fit together."
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

<style>
  /* Keep long equations scrollable within the article on narrow screens. */
  .page__content mjx-container[display="true"] {
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.25em 0;
  }
  .page__content figure img { display: block; width: 100%; height: auto; }
  .page__content figcaption { line-height: 1.55; }
</style>

**Research note.** This post organizes the current Sieve SGMM
method from my RA work with Yifan Hu. It brings together the algorithm,
basis-dependent identification, finite-time analysis, and existing simulation
results. The convergence statements concern a fixed finite basis and include
explicit conditions on the region visited by the iterates.

Instrumental variable regression has two problems to solve: use the
instrument to separate structural signal from endogenous noise, and do so
with an algorithm that can process a stream of observations. A good online
optimizer cannot recover parameter directions that the chosen moments never
see. Conversely, identifying moments do not automatically make a nonlinear
iteration converge from an arbitrary starting point.

Sieve SGMM, abbreviated **SSGMM**, connects these two questions. It replaces
an infinite family of conditional moment restrictions with a finite set of
instrument basis functions, then solves the resulting moments using a
running Jacobian and one new observation per update.

## Contents
{:.no_toc}

* TOC
{:toc}

## 1. From a conditional restriction to finite moments

Consider i.i.d. observations $$D_t=(z_t,x_t,y_t)$$ satisfying

$$
y=g(\theta_*;x)+\varepsilon,
\qquad \mathbb E[\varepsilon\mid z]=0,
\qquad \theta_*\in\mathbb R^d.
$$

The regressor $$x$$ may be correlated with $$\varepsilon$$, so minimizing
ordinary prediction error need not recover $$\theta_*$$. The instrument gives
us a different restriction: for every suitable function $$f$$,

$$
\mathbb E\!\left[f(z)\{g(\theta_*;x)-y\}\right]=0.
$$

Instead of estimating a first-stage model or a conditional expectation,
choose a finite vector of instrument functions
$$\psi_{\mathcal K}(z)\in\mathbb R^{p_{\mathcal K}}$$ and define

$$
\begin{aligned}
m_{\mathcal K}(D,\theta)
  &=\psi_{\mathcal K}(z)\{g(\theta;x)-y\},\\
M_{\mathcal K}(D,\theta)
  &=\psi_{\mathcal K}(z)\nabla_\theta g(\theta;x)^\top.
\end{aligned}
$$

Their expectations are $$\bar m_{\mathcal K}(\theta)$$ and
$$\bar M_{\mathcal K}(\theta)$$. The first is a $$p_{\mathcal K}$$-vector and the
second a $$p_{\mathcal K}\times d$$ matrix. We use the residual **$$g-y$$**
throughout, so $$D_\theta\bar m_{\mathcal K}=\bar M_{\mathcal K}$$ and a descent
step has a minus sign.

The identity-weighted population GMM objective is

$$
Q_{\mathcal K}(\theta)
 =\frac12\|\bar m_{\mathcal K}(\theta)\|^2,
\qquad
\nabla Q_{\mathcal K}(\theta)
 =\bar M_{\mathcal K}(\theta)^\top\bar m_{\mathcal K}(\theta).
$$

This is the squared norm of an **expected moment**. It differs from
$$\frac12\mathbb E\|m_{\mathcal K}(D,\theta)\|^2$$, whose gradient contains
same-observation products and generally targets a different objective.

### Why Hermite functions?

For standard Gaussian instruments, normalized multivariate Hermite
polynomials give orthonormal coordinates:

$$
\mathbb E[\psi_{\mathcal K}(z)\psi_{\mathcal K}(z)^\top]=I.
$$

In one dimension, the first four degree blocks are

$$
\psi_0(z)=1,\quad \psi_1(z)=z,\quad
\psi_2(z)=\frac{z^2-1}{\sqrt2},\quad
\psi_3(z)=\frac{z^3-3z}{\sqrt6}.
$$

In several dimensions, degree two also contains $$z_i z_j$$ for $$i<j$$.
Here $$\mathcal K$$ selects **degree blocks**, not individual coordinates.
For example, $$\mathcal K=\{0,1,2\}$$ has

$$
p_{\{0,1,2\}}=1+d_z+\frac{d_z(d_z+1)}2
$$

scalar moments. This distinction matters for both rank and computational
cost. Hermite orthonormality relies on the instrument distribution; it
cannot be assumed for arbitrary unnormalized or non-Gaussian data.

Orthonormality normalizes the basis, but it does not generally normalize
the moment noise. At the truth,

$$
\Omega_{\mathcal K}
 =\mathbb E\!\left[
   \psi_{\mathcal K}(z)\psi_{\mathcal K}(z)^\top\varepsilon^2
 \right].
$$

Only with additional conditions, such as constant conditional error
variance, is this proportional to $$I$$. Identity weighting is a deliberate
design choice here; efficiency of optimally weighted GMM does not follow
from basis orthonormality alone.

## 2. The online algorithm

Fix the basis and suppress its subscript. Let $$S_t$$ track the population
Jacobian and define the regularized gain

$$
A_\lambda(S)=(S^\top S+\lambda I)^{-1}S^\top,
\qquad \lambda>0.
$$

After receiving one observation $$D_{t+1}$$, use

$$
\begin{aligned}
A_t &= A_\lambda(S_t),\\
\theta_{t+1}
 &=\Pi_\Theta\!\left[
      \theta_t-\alpha_{t+1}A_t m(D_{t+1},\theta_t)
   \right],\\
S_{t+1}
 &=(1-\beta_{t+1})S_t+
      \beta_{t+1}M(D_{t+1},\theta_t),\\
\bar\theta_T&=\frac1T\sum_{t=1}^T\theta_t.
\end{aligned}
$$

The output is the Polyak–Ruppert average $$\bar\theta_T$$.
$$\Pi_\Theta$$ is Euclidean projection onto a specified compact convex
parameter set. The preconditioning design follows the nonlinear extension
of [SGMM by Chen et al.](https://doi.org/10.1093/jjfinec/nbad027).

The order of operations is essential. Both $$m$$ and $$M$$ use the same new
observation and the same old parameter $$\theta_t$$, but the parameter step
uses the **old state $$S_t$$**. The observation enters the Jacobian state only
after that step. With $$\mathcal F_t$$ denoting the history,

$$
\mathbb E[A_t m(D_{t+1},\theta_t)\mid\mathcal F_t]
  =A_t\bar m(\theta_t).
$$

There is no need to split the stream into two independent batches. This
conditional mean is still not generally $$\nabla Q(\theta_t)$$ or the exact
population Gauss–Newton direction: the gain uses an estimated, changing
Jacobian. Tracking that Jacobian is part of the analysis.

### Two time scales and a fixed ridge

Use step sizes

$$
\alpha_t=a(t+t_0)^{-\eta},\qquad
\beta_t=b(t+t_0)^{-\kappa},\qquad
\frac12<\kappa<\eta\le1.
$$

Because $$\alpha_t/\beta_t\to0$$, the Jacobian moves on the faster time scale.
It must adapt as the parameter moves. A simple cumulative average of all
past Jacobians can adapt too slowly when $$g$$ is nonlinear in $$\theta$$.

The ridge bounds the gain uniformly:

$$
\|A_\lambda(S)\|_{\mathrm{op}}\le\frac1{2\sqrt\lambda}.
$$

It regularizes the update, not the structural parameter. The method does
not add a $$\lambda\|\theta\|^2$$ penalty to the moment objective, and
$$\bar m(\theta_*)=0$$ remains a root. A fixed ridge nevertheless affects
contraction and finite-time constants.

For fixed $$p=p_{\mathcal K}$$, the persistent state uses $$O(pd+d)$$ storage.
A direct implementation forms a $$d\times d$$ Gram matrix and solves a
linear system each step, costing $$O(pd^2+d^3)$$ in addition to evaluating
the model and basis. Streaming memory does not imply a negligible cost
when many moments or parameters are used.

## 3. What does the chosen basis identify?

Define the conditional residual

$$
b_\theta(z)=\mathbb E[g(\theta;x)-y\mid z].
$$

Let $$\Pi_{\mathcal K}$$ be projection in $$L^2(P_z)$$ onto the selected
orthonormal basis. Then

$$
Q_{\mathcal K}(\theta)
  =\frac12\|\Pi_{\mathcal K}b_\theta\|_{L^2}^2.
$$

The finite moments see only the selected coordinates of the conditional
residual. They identify $$\theta_*$$ globally on a parameter set exactly when
every other parameter has a nonzero projected residual. Moment validity
at the truth, by itself, says nothing about uniqueness.

### The weakest visible parameter direction

The conditional sensitivity at the truth is
$$a_*(z)=\mathbb E[\nabla_\theta g(\theta_*;x)\mid z]$$. Write

$$
\begin{aligned}
C_k&=\mathbb E[\psi_k(z)a_*(z)^\top],\\
G_{\mathcal K}&=\bar M_{\mathcal K}(\theta_*),\\
J_{\mathcal K}&=G_{\mathcal K}^\top G_{\mathcal K}
            =\sum_{k\in\mathcal K}C_k^\top C_k,\\
\mu_{\mathcal K}&=\lambda_{\min}(J_{\mathcal K}).
\end{aligned}
$$

Equivalently,

$$
\mu_{\mathcal K}
 =\inf_{\|v\|=1}\|\Pi_{\mathcal K}[a_*^\top v]\|_{L^2}^2.
$$

Thus $$\mu_{\mathcal K}$$ is the signal in the least visible parameter
direction. If it is positive and $$\bar M_{\mathcal K}$$ is locally
$$L_{\mathcal K}$$-Lipschitz, Taylor expansion gives

$$
\|\bar m_{\mathcal K}(\theta_*+e)\|
 \ge\sqrt{\mu_{\mathcal K}}\|e\|
       -\frac{L_{\mathcal K}}2\|e\|^2.
$$

This yields regular local identification. The full-column-rank principle
is classical; see [Chen, Chernozhukov, Lee, and Newey
(2014)](https://doi.org/10.3982/ECTA9988). Having $$p_{\mathcal K}\ge d$$ is
necessary for full column rank, but does not ensure it. Rank deficiency
can sometimes permit higher-order identification, which the regular
convergence results below do not cover.

For a complete orthonormal system, define
$$J_\infty=\mathbb E[a_*a_*^\top]$$. If $$J_\infty\succ0$$, then

$$
\mu_{\mathcal K}
 \ge\lambda_{\min}(J_\infty)
       -\mathbb E\|a_*-\Pi_{\mathcal K}a_*\|^2.
$$

A finite basis therefore suffices for regular local identification once
its sensitivity approximation error is small enough. This is an
existence criterion; a useful numerical degree cutoff requires a bound
for the particular model.

### Example: linear instruments miss an even first stage

Take $$g(\theta;x)=\sin(\theta x)$$, $$\theta_*=0$$, and

$$
x=\gamma_*(z^2-1)+u,\qquad z\sim N(0,1),\qquad\gamma_*\ne0,
$$

where $$(u,\varepsilon)$$ is jointly centered Gaussian and independent of
$$z$$. The two noises may be correlated. Because the conditional law of
$$x$$ is unchanged by $$z\mapsto-z$$,

$$
\bar m_{\{1\}}(\theta)=\mathbb E[z\sin(\theta x)]=0
\quad\text{for every }\theta.
$$

No amount of optimization with the degree-one moment can distinguish
these parameters. The degree-two instrument instead gives

$$
G_{\{2\}}=\mathbb E\!\left[\frac{z^2-1}{\sqrt2}x\right]
 =\sqrt2\gamma_*,\qquad \mu_{\{2\}}=2\gamma_*^2>0.
$$

The quadratic moment recovers local sensitivity that the linear moment
misses. This example does not assert global uniqueness of the sine model.

### Example: full local rank can leave two roots

Now let $$x=z+u$$, with the same distributional assumptions, and put
$$c_x=\mathbb E[x^2]$$. Consider

$$
g(\theta;x)=\theta x+\theta^2(x^2-c_x).
$$

The conditional residual is

$$
b_\theta(z)=(\theta-\theta_*)z
             +(\theta^2-\theta_*^2)(z^2-1).
$$

Degree two alone gives
$$\bar m_{\{2\}}(\theta)=\sqrt2(\theta^2-\theta_*^2)$$. For
$$\theta_*\ne0$$, its derivative at the truth is nonzero, yet both
$$\theta_*$$ and $$-\theta_*$$ solve the moment. Degree one gives
$$\bar m_{\{1\}}(\theta)=\theta-\theta_*$$ and resolves the ambiguity.
The relevant question is which parameter directions a basis captures,
rather than its highest degree.

### Quadratic regression and missing coefficient directions

For the structural model $$g=b^\top x+x^\top A x$$, take $$A$$ symmetric and
$$x=\Gamma^\top z+u$$, where $$z$$ is standard Gaussian, $$u$$ is independent
with mean zero and covariance $$I$$, and $$\Gamma$$ has full column rank.
For parameter changes $$(\Delta_b,\Delta_A)$$,

$$
b_\theta(z)=\Delta_b^\top\Gamma^\top z
 +z^\top\Gamma\Delta_A\Gamma^\top z+\operatorname{tr}(\Delta_A).
$$

Degree one identifies $$b$$, while adding the constant supplies only one
weighted trace of $$A$$. Degrees $$\{1,2\}$$ identify all coefficients: degree
two determines $$\Gamma\Delta_A\Gamma^\top$$, and full column rank of
$$\Gamma$$ determines $$\Delta_A$$. Blocks of degree three or higher have
zero local sensitivity in this setting.

Although this regression is nonlinear in $$x$$, it is **linear in the
unknown coefficients $$(b,A)$$**. That distinction changes the convergence
analysis.

## 4. What convergence rate is justified?

The October 20 research note establishes sufficient finite-time bounds
for a fixed identifying basis. For nonlinear parameterizations, the
conditions include all of the following:

- A deterministic compact convex set $$\Theta$$ containing $$\theta_*$$,
  with initialization and all iterates in that set.
- A Lipschitz population Jacobian and uniform fourth-moment bounds for
  the sampled moment and centered sampled Jacobian on $$\Theta$$.
- Positive signal $$\mu_{\mathcal K}>0$$ and a small enough local region
  for the population update to point toward $$\theta_*$$.
- The two step-size schedules above, with an offset large enough to
  keep the early steps within the theorem's bounds.

For specificity, define

$$
\begin{aligned}
h_{\mathcal K}&=\frac{\mu_{\mathcal K}}{\mu_{\mathcal K}+\lambda},
&\ell_\lambda&=\frac5{4\lambda},\\
c_{\mathcal K}&=L_{\mathcal K}
 \left(\ell_\lambda\|G_{\mathcal K}\|_{\mathrm{op}}
           +\frac1{4\sqrt\lambda}\right).
\end{aligned}
$$

A sufficient region condition is
$$\Theta\subseteq B(\theta_*,r_{\mathcal K})$$ with
$$r_{\mathcal K}c_{\mathcal K}\le h_{\mathcal K}/4$$.
Projection enforces membership in a chosen set; it does not construct
this local set or locate the unknown truth. A large ball around zero
does not automatically satisfy this condition.

### General power schedules

Set

$$
\rho=\min\{\eta,\;2\kappa,\;4(\eta-\kappa)\}.
$$

Under the stated assumptions, when $$\eta<1$$,

$$
\mathbb E\|\theta_T-\theta_*\|^2=O(T^{-\rho}).
$$

When $$\eta=1$$, write $$\gamma_{\mathcal K}=a h_{\mathcal K}$$. The bound is

$$
\mathbb E\|\theta_T-\theta_*\|^2=
\begin{cases}
O(T^{-\rho}),&\gamma_{\mathcal K}>\rho,\\
O(T^{-\rho}\log T),&\gamma_{\mathcal K}=\rho,\\
O(T^{-\gamma_{\mathcal K}}),&\gamma_{\mathcal K}<\rho.
\end{cases}
$$

The averaged estimator has the same guaranteed orders. If
$$\eta+\rho>1$$, the iterates also converge almost surely.
All constants may depend on the fixed basis and the chosen region.

A convenient setting is

$$
\alpha_t=\frac a{t+t_0},\qquad
\beta_t=\frac b{(t+t_0)^{2/3}},\qquad a h_{\mathcal K}>1.
$$

It gives $$O(1/T)$$ **mean-square parameter error** for both the last
iterate and its average. The corresponding root-mean-square error is
$$O(T^{-1/2})$$. This statement does not establish asymptotic efficiency
or a central limit theorem.

If the same harmonic coefficient $$a$$ and ridge $$\lambda$$ are used across
bases, the sufficient $$1/T$$ condition becomes

$$
a>1\quad\text{and}\quad
\mu_{\mathcal K}>\frac\lambda{a-1}.
$$

An identifying but weak basis can therefore have a slower guaranteed
rate with the same coefficient. Adjusting $$a$$ separately to make
$$a h_{\mathcal K}>1$$ gives the same $$1/T$$ power across identifying bases,
while leaving different constants and admissible regions.

### Where the three exponents come from

The key tracking error is
$$\delta_t=S_t-\bar M(\theta_t)$$. The proof obtains

$$
\mathbb E\|\delta_t\|_F^4
 =O\!\left(t^{-2\kappa}+t^{-4(\eta-\kappa)}\right).
$$

The first term comes from sampling the Jacobian. The second comes from
following a moving population Jacobian. Controlling the probability of a
large tracking error leads to a parameter-error recursion of the form

$$
u_{t+1}\le(1-h_{\mathcal K}\alpha_{t+1})u_t
  +C_1\alpha_{t+1}^2
  +C_2\alpha_{t+1}\mathbb E\|\delta_t\|_F^4,
\qquad u_t=\mathbb E\|\theta_t-\theta_*\|^2.
$$

Parameter noise, Jacobian noise, and tracking lag produce the three terms
in $$\rho$$. The argument does not assume that the two updates use
independent observations.

For example, $$(\eta,\kappa)=(0.7,0.6)$$ yields the conservative nonlinear
bound $$O(T^{-0.4})$$, while $$(0.9,0.6)$$ yields $$O(T^{-0.9})$$. These are
upper bounds under the region and regularity assumptions, not fitted
slopes for the simulation curves.

If $$g(\theta;x)=q(x)^\top\theta$$, the population Jacobian is constant.
There is no moving-target term, the curvature radius restriction
disappears, and the analysis improves to $$O(T^{-\eta})$$ for $$\eta<1$$,
or $$O(1/T)$$ for $$\eta=1$$ with $$a h_{\mathcal K}>1$$, on the chosen compact
set. This includes the quadratic regression in coefficients above.

### Identification and convergence are separate conditions

Even global uniqueness of the moment root need not imply global
convergence of this iteration. For the deterministic moment
$$\bar m(\theta)=\theta^3-1$$, the unique root is $$1$$. Starting at
$$\theta_0=0$$, $$S_0=0$$ leaves both states at zero because the sampled
Jacobian vanishes there.

A global nonlinear result requires additional control, such as a
uniform alignment condition

$$
(\theta-\theta_*)^\top
 A_\lambda(\bar M(\theta))\bar m(\theta)
 \ge c_0\|\theta-\theta_*\|^2,
\qquad \theta\in\Theta,\quad c_0>0,
$$

together with regularity on that entire set. Local rank alone does not
provide it.

## 5. What the existing simulations show

The October 5 simulation note compares four degree sets:
$$H_1=\{1\}$$, $$H_{01}=\{0,1\}$$, $$H_{012}=\{0,1,2\}$$, and
$$H_{0123}=\{0,1,2,3\}$$. Each configuration was repeated ten times, with
$$10^7$$ updates. The recorded SSGMM settings were
$$\alpha_t=0.1(t+10)^{-0.7}$$, $$\beta_t=(t+10)^{-0.6}$$, and
$$\lambda=10^{-2}$$. TOSG used $$\alpha_t=0.1t^{-0.5}$$.

The following are selected **final median Euclidean parameter errors**
from that note, rather than mean-square rates:

- **Quadratic structure, linear first stage:** $$H_1$$ gives $$2.7821$$,
  $$H_{01}$$ gives $$1.1563$$, and $$H_{012}$$ gives $$0.0017$$.
  This is consistent with the missing quadratic coefficient directions
  in the smaller bases.
- **Quadratic structure, ReLU first stage:** the errors for $$H_{012}$$
  and $$H_{0123}$$ are $$0.0919$$ and $$0.0286$$. The extra cubic block helps
  in this recorded configuration.
- **Exponential structure, tanh first stage:** $$H_{012}$$ gives $$0.0095$$,
  whereas $$H_{0123}$$ gives $$0.2100$$. Adding a degree does not help
  uniformly across models.

<figure>
  <a href="{{ '/images/sieve-sgmm/quadratic-relu-iterations.png' | relative_url }}">
    <img src="{{ '/images/sieve-sgmm/quadratic-relu-iterations.png' | relative_url }}" alt="Historical quadratic–ReLU experiment: parameter error and prediction MSE against iterations for four SSGMM bases and TOSG." loading="lazy" width="1434" height="588">
  </a>
  <figcaption>Quadratic structure with a ReLU first stage. The stored curves illustrate how additional basis degrees can improve recovery. Click the figure to inspect its labels.</figcaption>
</figure>

<figure>
  <a href="{{ '/images/sieve-sgmm/exp-tanh-iterations.png' | relative_url }}">
    <img src="{{ '/images/sieve-sgmm/exp-tanh-iterations.png' | relative_url }}" alt="Historical exponential–tanh experiment: the degree-two sieve achieves a smaller final parameter error than the degree-three sieve." loading="lazy" width="1434" height="588">
  </a>
  <figcaption>Exponential structure with a tanh first stage. The same expansion to degree three can worsen recovery. Both figures are unchanged artifacts from the October 5 note; they have not been rerun with the current code.</figcaption>
</figure>

These plots compare equal **iteration counts**. SSGMM uses one ordinary
observation per step, while TOSG uses two conditional draws sharing an
instrument. The figures therefore do not establish equal-data-budget or
equal-oracle comparisons. The shaded regions are retained from the saved
plots; they are not interpreted here as confidence intervals.

Prediction MSE is also a different target from structural parameter
recovery. With endogenous noise, its cross term need not vanish away
from the truth, so a lower prediction MSE need not indicate a better IV
estimate. The parameter-error panels are the more direct diagnostic in
these simulations, where $$\theta_*$$ is known.

## 6. Reading the current implementation

The current `SSGMM` class fixes the Hermite family, Newton-type gain,
identity moment weighting, one observation per step, a running Jacobian,
parameter projection, and iterate averaging. The degree set is the main
structural choice; scalar step sizes and the ridge remain configurable.
The implementation currently supports degree blocks zero through three.

The experiment configuration inspected for this post uses

```python
ALGO_SSGMM_BASIS = [(1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
ALGO_SSGMM_LR = 0.1
ALGO_SSGMM_LR_DECAY = 0.7
ALGO_SSGMM_JACOBIAN_LR = 1.0
ALGO_SSGMM_JACOBIAN_LR_DECAY = 0.6
ALGO_SSGMM_LR_OFFSET = 10.0
ALGO_SSGMM_REG = 1e-2
ALGO_SSGMM_CLIP = 20.0
```

The compatibility name `CLIP` now denotes projection of the **parameter**
onto a ball of radius 20 about zero. It does not clip the random
direction. This stability device must be distinguished from the smaller
local region required by the nonlinear theorem. The October 20 note
still describes its simulations as unprojected; the implementation
inspected on October 23 already includes parameter projection.

The class defaults are different from the explicit experiment settings:
its parameter step is $$0.05(t+10)^{-0.8}$$, with the same Jacobian exponent
$$0.6$$. Under the nonlinear theorem's assumptions, these exponents give
$$\rho=0.8$$. Neither configuration is the harmonic schedule used for the
$$1/T$$ corollary.

At the level of one iteration, the method can be written as follows.
Here `project` is Euclidean projection onto the chosen parameter set,
and `grad_g` returns a $$d$$-vector.

```python
# theta, S, and theta_avg hold the state before this iteration.
# S has shape (p, d); each observation is used exactly once.
z, x, y = next(stream)
psi = basis(z)                         # shape (p,)
moment = psi * (g(theta, x) - y)
jacobian = np.outer(psi, grad_g(theta, x))

# The gain uses the old S, before absorbing the current observation.
direction = np.linalg.solve(S.T @ S + ridge * np.eye(d), S.T @ moment)
theta_new = project(theta - alpha(t + 1) * direction)
S_new = (1 - beta(t + 1)) * S + beta(t + 1) * jacobian
theta_avg = (t * theta_avg + theta_new) / (t + 1)
theta, S = theta_new, S_new
```

If $$S_0=0$$, the first parameter step is zero. The first observation
initializes the Jacobian state for the subsequent steps. Inverting the
Gram matrix explicitly is unnecessary; a linear solve gives the direction.

## 7. Choosing a basis and interpreting the scope

For nested orthonormal bases, $$J_{\mathcal K}$$ increases in
positive-semidefinite order. At the population level, retaining existing
blocks while adding new ones cannot reduce $$\mu_{\mathcal K}$$. But blocks
with $$C_k=0$$ add no signal, and estimated extra rows can add noise and
cost at finite time. Curvature and noise constants may also grow.

Basis choice should therefore proceed through three concrete questions:

1. Does the basis capture every parameter direction, or leave a rank
   deficiency or a global ambiguity?
2. Is its weakest signal strong enough relative to the ridge and step
   coefficient, and is the required nonlinear region credible?
3. Do extra moments improve recovery enough to justify their noise and
   computational cost under the available sample budget?

The theory here concerns a fixed finite-dimensional structural parameter
and a fixed basis. If that basis already identifies the correctly specified
parameter, truncation does not automatically create parameter bias:
$$\theta_*$$ still solves all selected valid moments. An insufficient basis
can instead leave multiple roots or invisible directions.

A growing or adaptively selected basis requires additional analysis,
including uniform control of signal, curvature, and noise as dimension
changes. The current results also do not give nonlinear convergence from
arbitrary initialization, rates for higher-order identification, or
efficient inference. Those questions remain separate from the finite-time
parameter bounds developed here.

## References and source note

- Chen, X., Lee, S., Liao, Y., Seo, M. H., Shin, Y., and Song, M. (2025).
  [SGMM: Stochastic Approximation to Generalized Method of Moments](https://doi.org/10.1093/jjfinec/nbad027).
  *Journal of Financial Econometrics*, 23(1), nbad027.
- Chen, X., Chernozhukov, V., Lee, S., and Newey, W. K. (2014).
  [Local Identification of Nonparametric and Semiparametric Models](https://doi.org/10.3982/ECTA9988).
  *Econometrica*, 82(2), 785–809.
- Chen, X., Roy, A., Hu, Y., and Balasubramanian, K. (2024).
  [Stochastic Optimization Algorithms for Instrumental Variable Regression with Streaming Data](https://arxiv.org/abs/2405.19463).
  *Advances in Neural Information Processing Systems*, 37, 26510–26542.
  Its oracle and model assumptions differ from those used here; comparing
  rate exponents alone would not compare the strength of the results.

The SSGMM bounds, examples, and simulation summaries above are drawn from
my research progress notes dated October 5 and October 20, 2026, checked
against the implementation on October 23, 2026. The displayed nonlinear
rates summarize the analysis in those notes; they are not presented as
theorems proved by the cited SGMM or streaming-IV papers. The historical
plots are included as exploratory evidence, without a new simulation run.
