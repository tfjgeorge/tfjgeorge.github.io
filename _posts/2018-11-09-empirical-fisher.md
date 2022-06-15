---
layout: post
title: What is the empirical Fisher ?
draft: false
categories: [note]
---
Some recent papers mention that they use the inverse of the "empirical Fisher" as a preconditioner. The main reason is its simplicity of use since it only requires gradients of the loss with respect to the parameters for each individual example. These are the same gradients as the ones we use to estimate our expected gradient when using SGD, as opposed to the true Fisher used in natural gradient, where the gradients that we need are gradients sampled from the distribution represented by our neural network.

The update using the "empirical Fisher" is:

$$
\begin{eqnarray*}
\theta & \leftarrow & \theta-\eta\left(\underbrace{\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\right]}_{C}+\epsilon\mathbf{I}\right)^{-1}\underbrace{\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right]}_{g}
\end{eqnarray*}
$$

Where $$g$$ is often estimated using its minibatch estimate, and $$C$$ is the (uncentered) covariance of the gradients, also estimated using a minibatch, or a running average. $$\eta$$ is the learning rate, and $$\epsilon$$ is a Tikhonov damping parameter.

## What problem are we solving when using this update?

<b>Claim</b>: This update is solution to the following problem, up to a second order approximation:

$$
\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right]=c
$$

Where we defined $$\Delta\ell\left(x,\theta\right)=\ell\left(x,\theta+\Delta\theta\right)-\ell\left(x,\theta\right)$$, and $$c$$ is a predefined scalar constant.

<i>Proof: </i>We start by writing the first order Taylor series expansion of $$\ell\left(x,\theta+\Delta\theta\right)$$:

$$
\begin{eqnarray*}
\Delta\ell\left(x,\theta\right) & = & \left(\ell\left(x,\theta\right)+\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)\right)-\ell\left(x,\theta\right)\\
 & = & \left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)
\end{eqnarray*}
$$

Where $$o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)$$ hides the higher order terms. It is a function such that $$\lim_{x\rightarrow0}\frac{o\left(x\right)}{x}=0$$, or to put it into words, it will be negligible compared to the first order term as long as $$\left\Vert \Delta\theta\right\Vert _{2}$$ is not too big.

By replacing in the constraint we obtain:

$$
\begin{eqnarray*}
\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right] & = & \mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)\right)^{2}\right]\\
 & = & \mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)^{2}\right]+o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
\end{eqnarray*}
$$

In the second line we have hidden the cross product in $$o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)$$.

We now remark that we can rewrite:

$$
\begin{eqnarray*}
\mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)^{2}\right] & = & \mathbb{E}\left[\left(\Delta\theta^{\top}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)\right)\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)\right]\\
 & = & \Delta\theta^{\top}\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\right]\Delta\theta\\
 & = & \Delta\theta^{\top}C\Delta\theta
\end{eqnarray*}
$$

And so our minimization problem becomes:

$$
\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\Delta\theta^{\top}C\Delta\theta=c
$$

Which can be solved e.g. using Lagrange multipliers, and we obtain the update:

$$
\begin{eqnarray*}
\Delta\theta^{*} & = & -\eta\left(C+\epsilon\mathbf{I}\right)^{-1}\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right]
\end{eqnarray*}
$$

Where $$\eta$$ is a scalar that we usually define as being the (constant) learning rate, but to be more precise it should be set so that the constraint $$\Delta\theta^{\top}C\Delta\theta=c$$ is enforced. The role of $$\epsilon$$ is to make sure that regardless of the spectrum of $$C$$, the update will not get too big, and make our second order approximation wrong.

## Discussion

What does it mean to be solving this minimization problem?

$$
\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right]=c
$$

First, it means that we measure progress in the space of our loss function. It has the desirable effect of making this update invariant by reparametrization of the network, as long as $$\epsilon$$ is kept small.

Second, it means that we will encourage all examples to have their loss reduced by a similar amount, on average $$\sqrt{c}$$. Is this something desirable or not ? I don’t know but I am open to your suggestions!


