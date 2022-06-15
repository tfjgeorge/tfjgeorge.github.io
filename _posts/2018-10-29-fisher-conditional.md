---
layout: post
title: How to compute the Fisher of a conditional when applying natural gradient to neural networks?
comments: true
lyx: true
draft: false 
categories: [note]
---
<p class="Unindented">
This short note aims at explaining how we come up with an expression for the Fisher Information Matrix in the context of the conditional distributions represented by neural networks.
</p>
<p class="Indented">
In neural networks, the so called natural gradient is a preconditioner for the gradient descent algorithm, where the update is regularized so that each update <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta
</script>
</span> of the values of the parameters <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span> will be measured using the <span class="MathJax_Preview"><script type="math/tex">
KL
</script>
</span> divergence. This has some interesting properties, such as the effect of making the update invariant to reparametrization of our neural network: more explanation to come in another blog post. The update is given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta_{nat}\theta & = & -\mathbf{F}_{\theta}^{-1}\mathbb{E}_{\left(x,y\right)\sim\mathcal{D}_{train}}\left[\nabla_{\theta}\left\{ -\log p_{\theta}\left(y|x\right)\right\} \right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
where:
</p>
<ul>
<li>
the expectation is taken using (discrete) samples <span class="MathJax_Preview"><script type="math/tex">
\left(x,y\right)
</script>
</span> of the training set <span class="MathJax_Preview"><script type="math/tex">
\mathcal{D}_{train}
</script>
</span>;
</li>
<li>
<span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span> is our neural network with <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> the input (e.g. the pixels of an image), and <span class="MathJax_Preview"><script type="math/tex">
y
</script>
</span> the output (e.g. the 10 coefficients of the softmax for MNIST where we have 10 classes = 10 digits);
</li>
<li>
we use the negative log likelihood as our loss function <span class="MathJax_Preview"><script type="math/tex">
-\log p_{\theta}\left(y|x\right)
</script>
</span>, and so <span class="MathJax_Preview"><script type="math/tex">
\nabla_{\theta}\left\{ -\log p_{\theta}\left(y|x\right)\right\} 
</script>
</span> is the gradient of our loss with respect to the parameters <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span>;
</li>
<li>
<span class="MathJax_Preview"><script type="math/tex">
\mathbf{F}_{\theta}
</script>
</span> is the Fisher Information Matrix (FIM) , defined as:
</li>

</ul>
<p class="Unindented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{z\sim p_{\theta}\left(z\right)}\left[\frac{\partial\log p_{\theta}\left(z\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(z\right)}{\partial\theta}\right)^{\top}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
The link between the <span class="MathJax_Preview"><script type="math/tex">
KL
</script>
</span> and the FIM resides in the fact that the FIM is the second order term of the Taylor series expansion of the <span class="MathJax_Preview"><script type="math/tex">
KL
</script>
</span>: For a distribution <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(z\right)
</script>
</span> it is given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
KL\left(p_{\theta}\left(z\right)\parallel p_{\theta+\Delta\theta}\left(z\right)\right) & = & \Delta\theta^{\top}\mathbf{F}_{\theta}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
where <span class="MathJax_Preview"><script type="math/tex">
o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
</script>
</span> is negligible compared to <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta^{\top}\mathbf{F}\Delta\theta
</script>
</span> when <span class="MathJax_Preview"><script type="math/tex">
\left\Vert \Delta\theta\right\Vert _{2}
</script>
</span> is small, the first order term is <span class="MathJax_Preview"><script type="math/tex">
0
</script>
</span>.
</p>
<p class="Indented">
This is the general definition for <span class="MathJax_Preview"><script type="math/tex">
\mathbf{F}_{\theta}
</script>
</span>, using a density <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(z\right)
</script>
</span>. But when applying this technique to train neural networks, we model the conditional <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span>. <b>So how do we apply this to neural networks training, i.e. for the conditional <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span>?</b>
</p>
<p class="Indented">
Here is my explanation.
</p>
<p class="Indented">
Instead of just considering <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span> we will use the joint probability <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y,x\right)=p_{\theta}\left(y|x\right)p\left(x\right)
</script>
</span>. We have introduced <span class="MathJax_Preview"><script type="math/tex">
p\left(x\right)
</script>
</span> which is the distribution over the inputs. If the task is image classification, this is the distribution of the natural images <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span>. Usually we do not have access to <span class="MathJax_Preview"><script type="math/tex">
p\left(x\right)
</script>
</span> explicitely, but instead we have samples from it, which are the images in our training set.
</p>
<p class="Indented">
By replacing <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(z\right)
</script>
</span> with <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(x,y\right)
</script>
</span> in the formula above, we can consider <span class="MathJax_Preview"><script type="math/tex">
KL\left(p_{\theta}\left(x,y\right)\parallel p_{\theta+\Delta\theta}\left(x,y\right)\right)
</script>
</span> and write the FIM for this joint distribution:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{\left(x,y\right)\sim p_{\theta}\left(x,y\right)}\left[\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta}\right)^{\top}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Next we replace the joint with the product of the marginal over <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> and the conditional in the derivative:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta} & = & \frac{\partial\log\left(p_{\theta}\left(y|x\right)p\left(x\right)\right)}{\partial\theta}\\
 & = & \frac{\partial\left(\log p_{\theta}\left(y|x\right)+\log p\left(x\right)\right)}{\partial\theta}\\
 & = & \frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}+\frac{\partial\log p\left(x\right)}{\partial\theta}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
and since <span class="MathJax_Preview"><script type="math/tex">
p\left(x\right)
</script>
</span> does not depend on <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span> then <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial\log p\left(x\right)}{\partial\theta}=0
</script>
</span>. This simplifies in:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta} & = & \frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Equivalently for the expectation, we can take the expectation in 2 steps:
</p>
<ol>
<li>
sample a <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> from our training distribution;
</li>
<li>
for this value of <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> compute <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span> then sample multiple points to estimate the expectation over <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}\left(y|x\right)
</script>
</span>. Here we also require multiple backprops to compute the gradients for each sample <span class="MathJax_Preview"><script type="math/tex">
y
</script>
</span>.
</li>

</ol>
<p class="Unindented">
Finally we get the desired formula:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{x\sim p\left(x\right),y\sim p_{\theta}\left(y|x\right)}\left[\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\right)^{\top}\right]\\
 & = & \mathbb{E}_{x\sim p\left(x\right)}\left[\mathbb{E}_{y\sim p_{\theta}\left(y|x\right)}\left[\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\right)^{\top}\right]\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
And so we get the FIM for a conditional distribution.
</p>

