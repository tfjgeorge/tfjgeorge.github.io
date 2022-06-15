---
layout: post
title: Demystifying Natural Neural Networks
comments: true
lyx: true
draft: true
categories: [note]
---
<!--starthtml-->



<span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> introduced a reparametrization for feedforward neural networks that gives a huge improvement in performance for optimizing neural networks. In this blog note I will summarize the main idea and introduce a new notation that clarifies the centering trick used in the paper.


## The FIM in the context of the Natural Gradient

### Fisher Information Matrix

The Fisher information matrix (FIM) is a tool well used in statistics. In the context of machine learning, and in particular deep learning, we use its inverse as a preconditioner for the gradient descent algorithm. In this section, we show how the FIM can be derived from the KL divergence and how we get a better &ldquo;natural&rdquo; gradient using this information. Let us first recall the definition of the KL divergence for 2 distributions $$p$$ and $$q$$:


$$
\begin{eqnarray*}
\text{KL}\left(p\parallel q\right) & = & \mathbb{E}_{p}\left[\log\left(\frac{p}{q}\right)\right]
\end{eqnarray*}
$$



It is a non-negative quantity that looks like a measure of how much $$q$$ differs from $$p$$. In particular, $$\text{KL}\left(p\parallel q\right)=0$$ when $$p=q$$. Note that it is not symmetric, so it can not directly be used as a metric.


The idea of the natural gradient is to use the KL divergence as a regularizer when doing gradient descent. We will denote by $$p_{\theta}$$ a parametric model and $$\Delta\theta$$ a change in its parameter values. $$\text{KL}\left(p_{\theta}\parallel p_{\theta+\Delta\theta}\right)$$ is used as our regularizer, so that each change $$\Delta\theta$$ gives the same change in the distribution space. Instead of using the full expression for $$\text{KL}\left(p_{\theta}\parallel p_{\theta+\Delta\theta}\right)$$ we will use its second order Taylor series around $$\theta$$ (for full derivation see for instance <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>):


$$
\begin{eqnarray*}
\text{KL}\left(p_{\theta}\Vert p_{\theta+\Delta\theta}\right) & = & \Delta\theta^{T}\mathbf{F}\Delta\theta+o(\left\Vert \Delta\theta\right\Vert _{2}^{2})
\end{eqnarray*}
$$



This expression exhibits the FIM which can now be used directly as a regularizer.

### (Natural) gradient descent

The usual gradient descent algorithm can be formulated as the minimization of the following expression:


$$
\begin{eqnarray*}
\Delta\theta & = & \text{argmin}_{\Delta\theta}\left\{ \Delta\theta^{T}\nabla\mathcal{L}+\frac{\epsilon}{2}\left\Vert \Delta\theta\right\Vert ^{2}\right\}
\end{eqnarray*}
$$



We use the notation $$\mathcal{L}$$ for the expectation of the loss function over the distribution of the data. This expression can be easily solved giving the expression $$\Delta\theta=-\frac{1}{\lambda}\nabla\mathcal{L}$$. The parameter $$\epsilon$$ is the inverse of the learning rate, and controls how much each parameter can change. We will now add a new regularizer using the FIM, and transform the minimization problem into:


$$
\begin{eqnarray*}
\Delta\theta & = & \text{argmin}_{\Delta\theta}\left\{ \Delta\theta^{T}\nabla\mathcal{L}+\frac{\epsilon}{2}\left\Vert \Delta\theta\right\Vert ^{2}+\frac{\lambda}{2}\Delta\theta^{T}\mathbf{F}\Delta\theta\right\}
\end{eqnarray*}
$$



We now constrain our gradient step to be small in term of change of parameter values, and also to be small in term of how much the resulting distribution changes. This expression can be solved to give $$\Delta\theta=\frac{1}{\lambda}\left(\mathbf{F}+\epsilon\mathbf{I}\right)^{-1}\nabla\mathcal{L}$$. This expression also gives an insight for the role of $$\lambda$$ and $$\epsilon$$, which control 2 different but related quantities expressed by our constraints. This new update is called the natural gradient.

## Factorizing the FIM for neural networks

### An expression for the FIM using jacobians

Interestingly, for the usual distributions expressed by neural networks, the FIM takes the following simple form as shown by <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>:


$$
\begin{eqnarray*}
\mathbf{F} & = & \mathbb{E}_{x\sim q}\left[\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}^{T}D\left(\boldsymbol{y}\left(x\right)\right)\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}\right]
\end{eqnarray*}
$$



The values for $$x$$ are drawn from the data generating distribution $$q$$. The notation $$\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}$$ is used for the jacobian of the output of the network (i.e. the probability expressed at a given $$x$$ : $$p\left(y\mid x\right)$$), with respect to the parameters. In other words, it measures how much the output of the network will change for a given $$x$$ if we change the parameters. $$D$$ is a diagonal matrix with non negative diagonal terms, and depends of the cost function used. For the quadratic loss it is the identity.

### Approximations to the FIM

The FIM is difficult to compute because of its size ($$n_{parameters}\times n_{parameters}$$) and because in general we do not have an expression for $$q$$ but only samples from a training dataset.


A first approximation that we can make is by ignoring the interactions between layers. In this case the FIM takes the form of a block diagonal matrix, where each block is a square matrix which has the size of the parameters of a layer. For a neural network with $$n_{layers}$$ layers this reduces the FIM into $$n_{layers}$$ smaller matrices. We will denote by $$\mathbf{F}_{i}$$ the block corresponding to layer $$i$$.


A second common approximation we make in practice is to use the empirical FIM for a training dataset of $$n$$ examples $$x_{i}$$: $$\mathbf{F}=\frac{1}{n}\sum_{i}\boldsymbol{J}_{\boldsymbol{y}\left(x_{i}\right)}^{T}D\left(\boldsymbol{y}\left(x_{i}\right)\right)\boldsymbol{J}_{\boldsymbol{y}\left(x_{i}\right)}$$.

### Factorization of the $$\mathbf{F}_{i}$$s using the Kronecker product

In the rest of this note, we will restrict our analysis to a multilayer perceptron. Each layer is parametrized using a weight matrix $$W$$ of size $$\left(\text{out}\times\text{in}\right)$$ and a bias vector $$b$$ of size $$\left(\text{out}\right)$$. A layer consists in a linear transformation and a non-linearity $$f$$ to give the hidden representation of the next layer:


$$
\begin{eqnarray*}
h_{l+1}=f_{l}\left(a_{l}\right) & \text{with} & a_{l}=W_{l}h_{l}+b_{l}
\end{eqnarray*}
$$



We will focus on a single layer, and drop the subscript $$l$$. In the following, we will also focus on a single example in the expectation for the FIM. Each individual example has its own jacobian with respect to the parameter. In other words, this jacobian measures how the output of the network changes for this example, if you move the parameter values.


The jacobians for parameters $$W$$ and $$b$$ are obtained using the chain rule for derivation: $$\boldsymbol{J}_{\boldsymbol{y}}^{W}=\boldsymbol{J}_{\boldsymbol{y}}^{a}\boldsymbol{J}_{a}^{W}$$ and $$\boldsymbol{J}_{\boldsymbol{y}}^{b}=\boldsymbol{J}_{\boldsymbol{y}}^{a}\boldsymbol{J}_{a}^{b}$$. The jacobian for the bias simplifies as $$\boldsymbol{J}_{a}^{b}=\mathbf{I}$$. For the weight matrix, it is a little bit more tricky. As $$W$$ is a matrix and not a vector, we can not express a jacobian matrix directly. We will have to make use of the Kronecker product and the $$\text{vec}$$ operator. We start from the linear relation $$a=Wh+b$$ and remark that $$a=\text{vec}\left(a\right)$$ since $$a$$ is a vector. We can now make use of the formula $$\text{vec}\left(AXB\right)=\left(B^{T}\otimes A\right)\text{vec}\left(X\right)$$ to obtain:


$$
\begin{eqnarray*}
a & = & \left(h^{T}\otimes\mathbf{I}_{out}\right)\text{vec}\left(W\right)\\
\boldsymbol{J}_{a}^{\text{vec}\left(W\right)} & = & \left(h^{T}\otimes\mathbf{I}_{out}\right)
\end{eqnarray*}
$$



Putting everything together we get the jacobians:


$$
\begin{eqnarray*}
\boldsymbol{J}_{\boldsymbol{y}}^{\text{vec}\left(W\right)} & = & \boldsymbol{J}_{\boldsymbol{y}}^{a}\left(h^{T}\otimes\mathbf{I}_{out}\right)\\
 & = & h^{T}\otimes\boldsymbol{J}_{\boldsymbol{y}}^{a}\\
\boldsymbol{J}_{\boldsymbol{y}}^{b} & = & \boldsymbol{J}_{\boldsymbol{y}}^{a}
\end{eqnarray*}
$$



Now imagine that we stack all parameters $$W$$ and $$b$$ in a vector $$\theta=\left(\text{vec}\left(W\right)_{1}\cdots\text{vec}\left(W\right)_{in\times out}b_{1}\cdots b_{out}\right)$$. We get the full jacobian by stacking the jacobians:


$$
\begin{eqnarray*}
\boldsymbol{J}_{\boldsymbol{y}}^{\theta} & = & \left(\begin{array}{cc}
h^{T} & 1\end{array}\right)\otimes\boldsymbol{J}_{\boldsymbol{y}}^{a}
\end{eqnarray*}
$$



From this expression we can finally express the FIM in a factorized form:


$$
\begin{eqnarray}
\mathbf{F} & = & \mathbb{E}\left[\left\{ \left(\begin{array}{c}
h\\
1
\end{array}\right)\otimes\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}\right\} D\left(\boldsymbol{y}\right)\left\{ \left(\begin{array}{cc}
h^{T} & 1\end{array}\right)\otimes\boldsymbol{J}_{\boldsymbol{y}}^{a}\right\} \right]\nonumber \\
 & = & \mathbb{E}\left[\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)\otimes\left(\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}D\left(\boldsymbol{y}\right)\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)\right]\label{eq:factoredfim}
\end{eqnarray}
$$



In this expression, remember that all 3 variables $$h$$, $$a$$ and $$\boldsymbol{y}$$ have a different value for each individual example. The FIM is the sum of the contributions of each example. The use of the Kronecker product permits splitting the contribution of each example in a term that involves the input of the layer $$\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)$$ and a term that involves the jacobian received on the output of the linear transformation $$\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}D\left(\boldsymbol{y}\right)\boldsymbol{J}_{\boldsymbol{y}}^{a}$$.

## KFAC

<span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> use a simplification of the FIM that drastically reduce the computation required for inverting it. Remember that even if we consider a block diagonal approximation of the FIM, each block still has size $$n_{parameters}\times n_{parameters}$$. We need to invert this block, and the operation of inverting a square matrix is $$O\left(n_{parameters}^{3}\right)$$. They propose the following approximation:


$$
\begin{eqnarray*}
\mathbf{F} & \approx & \mathbb{E}\left[\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)\right]\otimes\mathbb{E}\left[\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}D\left(\boldsymbol{y}\right)\boldsymbol{J}_{\boldsymbol{y}}^{a}\right]
\end{eqnarray*}
$$



The Kronecker product has the nice property that for 2 invertible square matrices $$A$$ and $$B$$, $$\left(A\otimes B\right)^{-1}=A^{-1}\otimes B^{-1}$$. It follows that inverting the FIM now requires inverting 2 smaller matrices. However the approximation they use is questionable, and they show some interesting experimental results.

## Natural Neural Networks

<span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> exploit the same factorization <a class="Reference" href="#eq:factoredfim">↓</a> but only consider the term involving the input of the linear transformation ($$h$$). They propose a reparametrization that will make $$\mathbb{E}\left[\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)\right]$$ equal the identity. To this view, they change the original linear transformation $$a=Wh+b$$ to become:


$$
\begin{eqnarray*}
a & = & VU\left(h-\mu\right)+d
\end{eqnarray*}
$$



$$V$$ is the new weight matrix and $$d$$ are the new biases. $$\mu=\mathbb{E}\left[h\right]$$ is the mean value for $$h$$ and $$U$$ is the square root of the inverse covariance of $$h$$, defined by $$U^{2}=\left(\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]\right)^{-1}$$, denoted by $$U=\left(\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]\right)^{-\frac{1}{2}}$$. $$U$$ and $$\mu$$ are not trained using gradient descent but instead they are estimated using data from the training set.


Our new parameters $$V$$ and $$d$$ are trained using gradient descent, which will now have the desired property. We will denote by $$h_{e}=U\left(h-\mu\right)$$ our new &ldquo;effective&rdquo; input to the linear transformation induced by the weight matrix $$V$$. Let us first remark that $$\mathbb{E}\left[h_{e}\right]=U\left(\mathbb{E}\left[h\right]-\mu\right)=U\left(\mu-\mu\right)=0$$, so the new input is centered on average. A second remark is that $$\mathbb{E}\left[h_{e}h_{e}^{T}\right]=U\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]U^{T}=\mathbf{I}$$. By construction $$U$$ cancels out the covariance. Wrapping everything together we thus have the desired property that:


$$
\begin{eqnarray*}
\mathbb{E}\left[\left(\begin{array}{cc}
h_{e}h_{e}^{T} & h_{e}\\
h_{e}^{T} & 1
\end{array}\right)\right] & = & \left(\begin{array}{cc}
\mathbb{E}\left[h_{e}h_{e}^{T}\right] & \mathbb{E}\left[h_{e}\right]\\
\mathbb{E}\left[h_{e}^{T}\right] & 1
\end{array}\right)\\
 & = & \left(\begin{array}{cc}
\mathbf{I} & 0\\
0 & 1
\end{array}\right)
\end{eqnarray*}
$$



The FIM for our new reparametrization thus has a better form, and they also show experimentally that this method is very efficient, and that by amortizing the cost of inverting the covariance matrix over several parameter updates, it can be faster that standard SGD.

## Conclusion

In this blog note we showed the path from the intuituion behind using the FIM as a preconditioner for gradient descent, to 2 recent algorithms that use the FIM. We also introduced the complete form for the FIM, also including the bias, which in the best of our knowledge has not been explicited in any publication at this time.


We think that the FIM or other similar preconditioners, and their factorizations, will open a new serie of algorithms to optimize neural networks, that will help make progress in artificial intelligence tasks.


## References

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-1"><span class="bib-index">1</span></a>] </span> <span class="bib-authors">Guillaume Desjardins, Karen Simonyan, Razvan Pascanu, others</span>: “<span class="bib-title">Natural neural networks</span>”, <i><span class="bib-booktitle">Advances in Neural Information Processing Systems</span></i>, pp. <span class="bib-pages">2071—2079</span>, <span class="bib-year">2015</span>.

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-2"><span class="bib-index">2</span></a>] </span> <span class="bib-authors">James Martens, Roger B Grosse</span>: “<span class="bib-title">Optimizing Neural Networks with Kronecker-factored Approximate Curvature.</span>”, <i><span class="bib-booktitle">ICML</span></i>, pp. <span class="bib-pages">2408—2417</span>, <span class="bib-year">2015</span>.

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-3"><span class="bib-index">3</span></a>] </span> <span class="bib-authors">Razvan Pascanu, Yoshua Bengio</span>: “<span class="bib-title">Revisiting natural gradient for deep networks</span>”, <i><span class="bib-journal">arXiv preprint arXiv:1301.3584</span></i>, <span class="bib-year">2013</span>.





<!--endhtml-->
