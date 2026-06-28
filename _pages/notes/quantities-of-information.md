---
layout: archive
title: "Quantities of Information"
permalink: /notes/quantities-of-information
author_profile: false
---

This blog introduces some fundamental concepts in information theory.


## Self-Information

The **self-information** (or **surprisal**) of an event $A$ is defined as:

$$
I(A) = -\log p(A),
$$

where $p(A)$ is the probability of the event $A$. This quantity represents the amount of information associated with the occurrence of $A$.

Intuitively, rarer events carry more information since they are more "surprising" to occur.
For example, let $X_i$ denote the extreme temperature event in a certain city during year $i$:

$$
X_i =
\begin{cases}
1, & \text{if an extreme heat event occurs},\\
0, & \text{if no extreme temperature event occurs},\\
-1, & \text{if an extreme cold event occurs}.
\end{cases}
$$

Suppose that the distribution of $X_i$ is given by

$$
P(X_i = k) =
\begin{cases}
0.2, & k = 1,\\
0.7, & k = 0,\\
0.1, & k = -1,
\end{cases}
$$

The self-information of the extreme temperature event is given by:

$$
I(X_i = k) = -\log p(X_i = k) \approx
\begin{cases}
2.322, & k = 1,\\
0.515, & k = 0,\\
3.322, & k = -1,
\end{cases}
$$

where the logarithm is calculated in base 2.
Hence, rare events bring more information.
When coding, this means that rare events should be assigned longer codewords to minimize the expected code length.


## Entropy

The **entropy** of a random variable $X$ is the expected value of the self-information:

$$
H(X) = \mathbb{E}_X[I(X)] = \mathbb{E}_X[-\log p(X)].
$$

Entropy measures the uncertainty associated with the random variable $X$.
It is maximized when all outcomes are equally likely, and in this case, the random variable is hard to predict.

For the extreme temperature example above, the entropy is calculated as:

$$
\begin{align*}
    H(X_i) &= -\sum_{k \in \{-1, 0, 1\}} p(X_i = k) \log p(X_i = k) \\
    &\approx 0.2 \times 2.322 + 0.7 \times 0.515 + 0.1 \times 3.322 \\
    &\approx 1.32.
\end{align*}
$$

This represents the average amount of information that would be obtained from observing the extreme temperature events.

In the coding theory, the entropy is also the expected code length when using optimal coding.
For a random variable $Y$ with uniform distribution over $n$ outcomes, its entropy is maximized and can be calculated by

$$
H(Y) = -\sum_{i=1}^{n} \frac{1}{n} \log \frac{1}{n} = \log n.
$$

Since $Y$ is uniformly distributed, we can arbitrarily assign codewords for the outcomes without worrying about the expected code length.
For example, when $n=4$, we can code each outcome with a unique 2-bit codeword (e.g., 00, 01, 10, 11), and the expected code length is exactly $\log_2 4 = 2$ bits.

>One may attempt to encode four equally likely outcomes using codewords {0,1,00,11}, resulting in an average code length of only 1.5 bits. 
>However, this code is not uniquely decodable because the bit string 00 may represent either the third outcome or two consecutive occurrences of the first outcome. 
>Hence, meaningful compression limits must be discussed only within the class of uniquely decodable codes.


## Joint Entropy
The **joint entropy** of two random variables $X$ and $Y$ is merely the entropy of the random vector $(X, Y)$:

$$
H(X, Y) = \mathbb{E}_{X,Y}[I(X, Y)] = \mathbb{E}_{X,Y}[- \log p(X,Y)].
$$

This quantity measures the uncertainty associated with the pair of random variables $(X, Y)$.
It should be noted that for independent random variables $X$ and $Y$, the joint entropy satisfies 

$$H(X, Y) = H(X) + H(Y).$$


## Conditional Entropy
The **conditional entropy** of $X$ given $Y$ is the expected value of the negative loglikelihood of $X$ given $Y$:

$$
H(X|Y) = \mathbb{E}_{X,Y}[-\log p(X|Y)].
$$

This quantity measures the remaining uncertainty in $X$ after observing $Y$.
It satisfies the relation $H(X|Y) = H(X, Y) - H(Y)$.


## Cross Entropy
The **cross entropy** between two distributions $p$ and $q$ is defined by

$$
C(p,q) = \mathbb{E}_p \left[ - \log q(X) \right],
$$
where we denote the cross entropy as $C$ instead of $H$ to eliminate the ambiguity with the joint entropy.
It is the expected number of bits needed to compress some data samples drawn from distribution $p$, but using a coding scheme that is optimized for distribution $q$. 



## Relative Entropy (Kullback-Leibler Divergence)

The **relative entropy** (or **Kullback-Leibler divergence**, **information gain**) is 

$$
\text{KL}(p \Vert q) = \mathbb{E}_{p}\left[\log \frac{p(X)}{q(X)}\right].
$$

Intuitively, it can be derived as

$$
\begin{align*}
    \text{KL}(p \Vert q) &= \mathbb{E}_{p}\left[\log \frac{p(X)}{q(X)}\right] \\
    &= -\mathbb{E}_p [-\log p(X)] + \mathbb{E}_p [-\log q(X)] \\
    &= C(p,q) - H(p) \\
    &= \text{Expected code length using $q$} - \text{Optimal code length with $p$}.
\end{align*}
$$

Hence, it can be interpreted as the *extra number of bits* you need to pay when coding data samples from distribution $p$ using an inaccurate coding scheme for distribution $q$.

It can be demonstrated that $\text{KL}(p \Vert q) \geq 0$, since

$$
\begin{align*}
    \text{KL}(p \Vert q) &= \mathbb{E}_p \left[\log \frac{p(X)}{q(X)}\right] \\
    &= \mathbb{E}_p \left[ - \log \frac{q(X)}{p(X)}\right] \\
    &\geq - \log \mathbb{E}_p \left[\frac{q(X)}{p(X)}\right] \\
    &= - \log \int \frac{q(x)}{p(x)} p(x) \, \text{d}x \\
    &= - \log 1 \\
    &= 0,
\end{align*}
$$

where the inequality follows from Jensen's inequality (by the convexity of the negative logarithm function).
The equality holds if and only if $p = q$, since in this case $\frac{q(x)}{p(x)}$ is linear and satisfies the equality condition for Jensen's inequality.
This result means that you always need to pay extra bits if the coding scheme is suboptimal.

Additionally, it is important to note that the KL divergence is not symmetric: $\text{KL}(p \Vert q) \neq \text{KL}(q \Vert p)$ in general, and does not satisfy the triangle inequality.
Therefore, although it can be used as a measure of dissimilarity between probability distributions, it should not be treated as a true metric distance.
