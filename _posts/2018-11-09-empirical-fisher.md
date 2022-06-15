---
layout: post
title: What is the empirical Fisher ?
comments: true
lyx: true
draft: false
categories: [note]
---
<p class="Indented">
Some recent papers mention that they use the inverse of the "empirical Fisher" as a preconditioner. The main reason is its simplicity of use since it only requires gradients of the loss with respect to the parameters for each individual example. These are the same gradients as the ones we use to estimate our expected gradient when using SGD, as opposed to the true Fisher used in natural gradient, where the gradients that we need are gradients sampled from the distribution represented by our neural network.
</p>
<p class="Indented">
The update using the "empirical Fisher" is:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\theta & \leftarrow & \theta-\eta\left(\underbrace{\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\right]}_{C}+\epsilon\mathbf{I}\right)^{-1}\underbrace{\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right]}_{g}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Where <span class="MathJax_Preview"><script type="math/tex">
g
</script>
</span> is often estimated using its minibatch estimate, and <span class="MathJax_Preview"><script type="math/tex">
C
</script>
</span> is the (uncentered) covariance of the gradients, also estimated using a minibatch, or a running average. <span class="MathJax_Preview"><script type="math/tex">
\eta
</script>
</span> is the learning rate, and <span class="MathJax_Preview"><script type="math/tex">
\epsilon
</script>
</span> is a Tikhonov damping parameter.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-1">1</a> What problem are we solving when using this update?
</h1>
<p class="Unindented">
<b>Claim</b>: This update is solution to the following problem, up to a second order approximation:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">

\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right]=c

</script>

</span>

</p>
<p class="Indented">
Where we defined <span class="MathJax_Preview"><script type="math/tex">
\Delta\ell\left(x,\theta\right)=\ell\left(x,\theta+\Delta\theta\right)-\ell\left(x,\theta\right)
</script>
</span>, and <span class="MathJax_Preview"><script type="math/tex">
c
</script>
</span> is a predefined scalar constant.
</p>
<p class="Indented">
<i>Proof: </i>We start by writing the first order Taylor series expansion of <span class="MathJax_Preview"><script type="math/tex">
\ell\left(x,\theta+\Delta\theta\right)
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta\ell\left(x,\theta\right) & = & \left(\ell\left(x,\theta\right)+\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)\right)-\ell\left(x,\theta\right)\\
 & = & \left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Where <span class="MathJax_Preview"><script type="math/tex">
o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)
</script>
</span> hides the higher order terms. It is a function such that <span class="MathJax_Preview"><script type="math/tex">
\lim_{x\rightarrow0}\frac{o\left(x\right)}{x}=0
</script>
</span>, or to put it into words, it will be negligible compared to the first order term as long as <span class="MathJax_Preview"><script type="math/tex">
\left\Vert \Delta\theta\right\Vert _{2}
</script>
</span> is not too big.
</p>
<p class="Indented">
By replacing in the constraint we obtain:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right] & = & \mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}\right)\right)^{2}\right]\\
 & = & \mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)^{2}\right]+o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
In the second line we have hidden the cross product in <span class="MathJax_Preview"><script type="math/tex">
o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
</script>
</span>.
</p>
<p class="Indented">
We now remark that we can rewrite:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbb{E}\left[\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)^{2}\right] & = & \mathbb{E}\left[\left(\Delta\theta^{\top}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)\right)\left(\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\Delta\theta\right)\right]\\
 & = & \Delta\theta^{\top}\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\left(\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right)^{\top}\right]\Delta\theta\\
 & = & \Delta\theta^{\top}C\Delta\theta
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
And so our minimization problem becomes:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">

\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\Delta\theta^{\top}C\Delta\theta=c

</script>

</span>

</p>
<p class="Indented">
Which can be solved e.g. using Lagrange multipliers, and we obtain the update:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta\theta^{*} & = & -\eta\left(C+\epsilon\mathbf{I}\right)^{-1}\mathbb{E}\left[\frac{\partial\ell\left(x,\theta\right)}{\partial\theta}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Where <span class="MathJax_Preview"><script type="math/tex">
\eta
</script>
</span> is a scalar that we usually define as being the (constant) learning rate, but to be more precise it should be set so that the constraint <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta^{\top}C\Delta\theta=c
</script>
</span> is enforced. The role of <span class="MathJax_Preview"><script type="math/tex">
\epsilon
</script>
</span> is to make sure that regardless of the spectrum of <span class="MathJax_Preview"><script type="math/tex">
C
</script>
</span>, the update will not get too big, and make our second order approximation wrong.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-2">2</a> Discussion
</h1>
<p class="Unindented">
What does it mean to be solving this minimization problem?
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">

\text{min}_{\Delta\theta}L\left(\theta+\Delta\theta\right)\text{ such that }\mathbb{E}\left[\left(\Delta\ell\left(x,\theta\right)\right)^{2}\right]=c

</script>

</span>

</p>
<p class="Indented">
First, it means that we measure progress in the space of our loss function. It has the desirable effect of making this update invariant by reparametrization of the network, as long as <span class="MathJax_Preview"><script type="math/tex">
\epsilon
</script>
</span> is kept small.
</p>
<p class="Indented">
Second, it means that we will encourage all examples to have their loss reduced by a similar amount, on average <span class="MathJax_Preview"><script type="math/tex">
\sqrt{c}
</script>
</span>. Is this something desirable or not ? I don’t know but I am open to your suggestions!
</p>

