---
layout: post
title: How to compute the Fisher of a conditional when applying natural gradient to neural networks?
comments: true
lyx: true
draft: false 
categories: [note]
---

This short note aims at explaining how we come up with an expression for the Fisher Information Matrix in the context of the conditional distributions represented by neural networks.


In neural networks, the so called natural gradient is a preconditioner for the gradient descent algorithm, where the update is regularized so that each update $$\Delta\theta$$ of the values of the parameters $$\theta$$ will be measured using the $$KL$$ divergence. This has some interesting properties, such as the effect of making the update invariant to reparametrization of our neural network: more explanation to come in another blog post. The update is given by:

$$
\begin{eqnarray*}
\Delta_{nat}\theta & = & -\mathbf{F}_{\theta}^{-1}\mathbb{E}_{\left(x,y\right)\sim\mathcal{D}_{train}}\left[\nabla_{\theta}\left\{ -\log p_{\theta}\left(y|x\right)\right\} \right]
\end{eqnarray*}
$$

where:

<ul>
<li>
the expectation is taken using (discrete) samples \(\left(x,y\right)\) of the training set \(\mathcal{D}_{train}\);
</li>
<li>
\(p_{\theta}\left(y|x\right)\) is our neural network with \(x\) the input (e.g. the pixels of an image), and \(y\) the output (e.g. the 10 coefficients of the softmax for MNIST where we have 10 classes = 10 digits);
</li>
<li>
we use the negative log likelihood as our loss function \(-\log p_{\theta}\left(y|x\right)\), and so \(\nabla_{\theta}\left\{ -\log p_{\theta}\left(y|x\right)\right\} \) is the gradient of our loss with respect to the parameters \(\theta\);
</li>
<li>
\(\mathbf{F}_{\theta}\) is the Fisher Information Matrix (FIM) , defined as:
</li>

</ul>

$$
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{z\sim p_{\theta}\left(z\right)}\left[\frac{\partial\log p_{\theta}\left(z\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(z\right)}{\partial\theta}\right)^{\top}\right]
\end{eqnarray*}
$$

The link between the $$KL$$ and the FIM resides in the fact that the FIM is the second order term of the Taylor series expansion of the $$KL$$: For a distribution $$p_{\theta}\left(z\right)$$ it is given by:


$$
\begin{eqnarray*}
KL\left(p_{\theta}\left(z\right)\parallel p_{\theta+\Delta\theta}\left(z\right)\right) & = & \Delta\theta^{\top}\mathbf{F}_{\theta}\Delta\theta+o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)
\end{eqnarray*}
$$

where $$o\left(\left\Vert \Delta\theta\right\Vert _{2}^{2}\right)$$ is negligible compared to $$\Delta\theta^{\top}\mathbf{F}\Delta\theta$$ when $$\left\Vert \Delta\theta\right\Vert _{2}$$ is small, the first order term is $$0$$.


This is the general definition for $$\mathbf{F}_{\theta}$$, using a density $$p_{\theta}\left(z\right)$$. But when applying this technique to train neural networks, we model the conditional $$p_{\theta}\left(y\vert x\right)$$. So how do we apply this to neural networks training, i.e. for the conditional $$p_{\theta}\left(y\vert x\right)$$?

Here is my explanation.


Instead of just considering $$p_{\theta}\left(y\vert x\right)$$ we will use the joint probability $$p_{\theta}\left(y,x\right)=p_{\theta}\left(y\vert x\right)p\left(x\right)$$. We have introduced $$p\left(x\right)$$ which is the distribution over the inputs. If the task is image classification, this is the distribution of the natural images $$x$$. Usually we do not have access to $$p\left(x\right)$$ explicitely, but instead we have samples from it, which are the images in our training set.


By replacing $$p_{\theta}\left(z\right)$$ with $$p_{\theta}\left(x,y\right)$$ in the formula above, we can consider $$KL\left(p_{\theta}\left(x,y\right)\parallel p_{\theta+\Delta\theta}\left(x,y\right)\right)$$ and write the FIM for this joint distribution:


$$
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{\left(x,y\right)\sim p_{\theta}\left(x,y\right)}\left[\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta}\right)^{\top}\right]
\end{eqnarray*}
$$

Next we replace the joint with the product of the marginal over $$x$$ and the conditional in the derivative:


$$
\begin{eqnarray*}
\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta} & = & \frac{\partial\log\left(p_{\theta}\left(y|x\right)p\left(x\right)\right)}{\partial\theta}\\
 & = & \frac{\partial\left(\log p_{\theta}\left(y|x\right)+\log p\left(x\right)\right)}{\partial\theta}\\
 & = & \frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}+\frac{\partial\log p\left(x\right)}{\partial\theta}
\end{eqnarray*}
$$


and since $$p\left(x\right)$$ does not depend on $$\theta$$ then $$\frac{\partial\log p\left(x\right)}{\partial\theta}=0$$. This simplifies in:


$$
\begin{eqnarray*}
\frac{\partial\log p_{\theta}\left(x,y\right)}{\partial\theta} & = & \frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}
\end{eqnarray*}
$$

Equivalently for the expectation, we can take the expectation in 2 steps:

<ol>
<li>
sample a \(x\) from our training distribution;
</li>
<li>
for this value of \(x\) compute \(p_{\theta}\left(y|x\right)\) then sample multiple points to estimate the expectation over \(p_{\theta}\left(y|x\right)\). Here we also require multiple backprops to compute the gradients for each sample \(y\).
</li>

</ol>

Finally we get the desired formula:

$$
\begin{eqnarray*}
\mathbf{F}_{\theta} & = & \mathbb{E}_{x\sim p\left(x\right),y\sim p_{\theta}\left(y|x\right)}\left[\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\right)^{\top}\right]\\
 & = & \mathbb{E}_{x\sim p\left(x\right)}\left[\mathbb{E}_{y\sim p_{\theta}\left(y|x\right)}\left[\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\left(\frac{\partial\log p_{\theta}\left(y|x\right)}{\partial\theta}\right)^{\top}\right]\right]
\end{eqnarray*}
$$

And so we get the FIM for a conditional distribution.


