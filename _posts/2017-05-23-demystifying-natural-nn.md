---
layout: post
title: Demystifying Natural Neural Networks
comments: true
lyx: true
draft: true
categories: [note]
---
<!--starthtml-->
<h1 class="title">
Demystifying Natural Neural Networks
</h1>
<h2 class="author">
Thomas George
</h2>
<p class="Unindented">
<span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> introduced a reparametrization for feedforward neural networks that gives a huge improvement in performance for optimizing neural networks. In this blog note I will summarize the main idea and introduce a new notation that clarifies the centering trick used in the paper.
</p>
<h1 class="Section">
The FIM in the context of the Natural Gradient
</h1>
<h2 class="Subsection">
Fisher Information Matrix
</h2>
<p class="Unindented">
The Fisher information matrix (FIM) is a tool well used in statistics. In the context of machine learning, and in particular deep learning, we use its inverse as a preconditioner for the gradient descent algorithm. In this section, we show how the FIM can be derived from the KL divergence and how we get a better &ldquo;natural&rdquo; gradient using this information. Let us first recall the definition of the KL divergence for 2 distributions <span class="MathJax_Preview"><script type="math/tex">
p
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
q
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\text{KL}\left(p\parallel q\right) & = & \mathbb{E}_{p}\left[\log\left(\frac{p}{q}\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
It is a non-negative quantity that looks like a measure of how much <span class="MathJax_Preview"><script type="math/tex">
q
</script>
</span> differs from <span class="MathJax_Preview"><script type="math/tex">
p
</script>
</span>. In particular, <span class="MathJax_Preview"><script type="math/tex">
\text{KL}\left(p\parallel q\right)=0
</script>
</span> when <span class="MathJax_Preview"><script type="math/tex">
p=q
</script>
</span>. Note that it is not symmetric, so it can not directly be used as a metric.
</p>
<p class="Indented">
The idea of the natural gradient is to use the KL divergence as a regularizer when doing gradient descent. We will denote by <span class="MathJax_Preview"><script type="math/tex">
p_{\theta}
</script>
</span> a parametric model and <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta
</script>
</span> a change in its parameter values. <span class="MathJax_Preview"><script type="math/tex">
\text{KL}\left(p_{\theta}\parallel p_{\theta+\Delta\theta}\right)
</script>
</span> is used as our regularizer, so that each change <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta
</script>
</span> gives the same change in the distribution space. Instead of using the full expression for <span class="MathJax_Preview"><script type="math/tex">
\text{KL}\left(p_{\theta}\parallel p_{\theta+\Delta\theta}\right)
</script>
</span> we will use its second order Taylor series around <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span> (for full derivation see for instance <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>):
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\text{KL}\left(p_{\theta}\Vert p_{\theta+\Delta\theta}\right) & = & \Delta\theta^{T}\mathbf{F}\Delta\theta+o(\left\Vert \Delta\theta\right\Vert _{2}^{2})
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
This expression exhibits the FIM which can now be used directly as a regularizer.
</p>
<h2 class="Subsection">
(Natural) gradient descent
</h2>
<p class="Unindented">
The usual gradient descent algorithm can be formulated as the minimization of the following expression:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta\theta & = & \text{argmin}_{\Delta\theta}\left\{ \Delta\theta^{T}\nabla\mathcal{L}+\frac{\epsilon}{2}\left\Vert \Delta\theta\right\Vert ^{2}\right\}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We use the notation <span class="MathJax_Preview"><script type="math/tex">
\mathcal{L}
</script>
</span> for the expectation of the loss function over the distribution of the data. This expression can be easily solved giving the expression <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta=-\frac{1}{\lambda}\nabla\mathcal{L}
</script>
</span>. The parameter <span class="MathJax_Preview"><script type="math/tex">
\epsilon
</script>
</span> is the inverse of the learning rate, and controls how much each parameter can change. We will now add a new regularizer using the FIM, and transform the minimization problem into:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta\theta & = & \text{argmin}_{\Delta\theta}\left\{ \Delta\theta^{T}\nabla\mathcal{L}+\frac{\epsilon}{2}\left\Vert \Delta\theta\right\Vert ^{2}+\frac{\lambda}{2}\Delta\theta^{T}\mathbf{F}\Delta\theta\right\}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We now constrain our gradient step to be small in term of change of parameter values, and also to be small in term of how much the resulting distribution changes. This expression can be solved to give <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta=\frac{1}{\lambda}\left(\mathbf{F}+\epsilon\mathbf{I}\right)^{-1}\nabla\mathcal{L}
</script>
</span>. This expression also gives an insight for the role of <span class="MathJax_Preview"><script type="math/tex">
\lambda
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\epsilon
</script>
</span>, which control 2 different but related quantities expressed by our constraints. This new update is called the natural gradient.
</p>
<h1 class="Section">
Factorizing the FIM for neural networks
</h1>
<h2 class="Subsection">
An expression for the FIM using jacobians
</h2>
<p class="Unindented">
Interestingly, for the usual distributions expressed by neural networks, the FIM takes the following simple form as shown by <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbf{F} & = & \mathbb{E}_{x\sim q}\left[\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}^{T}D\left(\boldsymbol{y}\left(x\right)\right)\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
The values for <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> are drawn from the data generating distribution <span class="MathJax_Preview"><script type="math/tex">
q
</script>
</span>. The notation <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{J}_{\boldsymbol{y}\left(x\right)}
</script>
</span> is used for the jacobian of the output of the network (i.e. the probability expressed at a given <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> : <span class="MathJax_Preview"><script type="math/tex">
p\left(y\mid x\right)
</script>
</span>), with respect to the parameters. In other words, it measures how much the output of the network will change for a given <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> if we change the parameters. <span class="MathJax_Preview"><script type="math/tex">
D
</script>
</span> is a diagonal matrix with non negative diagonal terms, and depends of the cost function used. For the quadratic loss it is the identity.
</p>
<h2 class="Subsection">
Approximations to the FIM
</h2>
<p class="Unindented">
The FIM is difficult to compute because of its size (<span class="MathJax_Preview"><script type="math/tex">
n_{parameters}\times n_{parameters}
</script>
</span>) and because in general we do not have an expression for <span class="MathJax_Preview"><script type="math/tex">
q
</script>
</span> but only samples from a training dataset.
</p>
<p class="Indented">
A first approximation that we can make is by ignoring the interactions between layers. In this case the FIM takes the form of a block diagonal matrix, where each block is a square matrix which has the size of the parameters of a layer. For a neural network with <span class="MathJax_Preview"><script type="math/tex">
n_{layers}
</script>
</span> layers this reduces the FIM into <span class="MathJax_Preview"><script type="math/tex">
n_{layers}
</script>
</span> smaller matrices. We will denote by <span class="MathJax_Preview"><script type="math/tex">
\mathbf{F}_{i}
</script>
</span> the block corresponding to layer <span class="MathJax_Preview"><script type="math/tex">
i
</script>
</span>.
</p>
<p class="Indented">
A second common approximation we make in practice is to use the empirical FIM for a training dataset of <span class="MathJax_Preview"><script type="math/tex">
n
</script>
</span> examples <span class="MathJax_Preview"><script type="math/tex">
x_{i}
</script>
</span>: <span class="MathJax_Preview"><script type="math/tex">
\mathbf{F}=\frac{1}{n}\sum_{i}\boldsymbol{J}_{\boldsymbol{y}\left(x_{i}\right)}^{T}D\left(\boldsymbol{y}\left(x_{i}\right)\right)\boldsymbol{J}_{\boldsymbol{y}\left(x_{i}\right)}
</script>
</span>.
</p>
<h2 class="Subsection">
Factorization of the <span class="MathJax_Preview"><script type="math/tex">
\mathbf{F}_{i}
</script>
</span>s using the Kronecker product
</h2>
<p class="Unindented">
In the rest of this note, we will restrict our analysis to a multilayer perceptron. Each layer is parametrized using a weight matrix <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> of size <span class="MathJax_Preview"><script type="math/tex">
\left(\text{out}\times\text{in}\right)
</script>
</span> and a bias vector <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span> of size <span class="MathJax_Preview"><script type="math/tex">
\left(\text{out}\right)
</script>
</span>. A layer consists in a linear transformation and a non-linearity <span class="MathJax_Preview"><script type="math/tex">
f
</script>
</span> to give the hidden representation of the next layer:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
h_{l+1}=f_{l}\left(a_{l}\right) & \text{with} & a_{l}=W_{l}h_{l}+b_{l}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We will focus on a single layer, and drop the subscript <span class="MathJax_Preview"><script type="math/tex">
l
</script>
</span>. In the following, we will also focus on a single example in the expectation for the FIM. Each individual example has its own jacobian with respect to the parameter. In other words, this jacobian measures how the output of the network changes for this example, if you move the parameter values.
</p>
<p class="Indented">
The jacobians for parameters <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span> are obtained using the chain rule for derivation: <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{J}_{\boldsymbol{y}}^{W}=\boldsymbol{J}_{\boldsymbol{y}}^{a}\boldsymbol{J}_{a}^{W}
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{J}_{\boldsymbol{y}}^{b}=\boldsymbol{J}_{\boldsymbol{y}}^{a}\boldsymbol{J}_{a}^{b}
</script>
</span>. The jacobian for the bias simplifies as <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{J}_{a}^{b}=\mathbf{I}
</script>
</span>. For the weight matrix, it is a little bit more tricky. As <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> is a matrix and not a vector, we can not express a jacobian matrix directly. We will have to make use of the Kronecker product and the <span class="MathJax_Preview"><script type="math/tex">
\text{vec}
</script>
</span> operator. We start from the linear relation <span class="MathJax_Preview"><script type="math/tex">
a=Wh+b
</script>
</span> and remark that <span class="MathJax_Preview"><script type="math/tex">
a=\text{vec}\left(a\right)
</script>
</span> since <span class="MathJax_Preview"><script type="math/tex">
a
</script>
</span> is a vector. We can now make use of the formula <span class="MathJax_Preview"><script type="math/tex">
\text{vec}\left(AXB\right)=\left(B^{T}\otimes A\right)\text{vec}\left(X\right)
</script>
</span> to obtain:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
a & = & \left(h^{T}\otimes\mathbf{I}_{out}\right)\text{vec}\left(W\right)\\
\boldsymbol{J}_{a}^{\text{vec}\left(W\right)} & = & \left(h^{T}\otimes\mathbf{I}_{out}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Putting everything together we get the jacobians:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\boldsymbol{J}_{\boldsymbol{y}}^{\text{vec}\left(W\right)} & = & \boldsymbol{J}_{\boldsymbol{y}}^{a}\left(h^{T}\otimes\mathbf{I}_{out}\right)\\
 & = & h^{T}\otimes\boldsymbol{J}_{\boldsymbol{y}}^{a}\\
\boldsymbol{J}_{\boldsymbol{y}}^{b} & = & \boldsymbol{J}_{\boldsymbol{y}}^{a}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Now imagine that we stack all parameters <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span> in a vector <span class="MathJax_Preview"><script type="math/tex">
\theta=\left(\text{vec}\left(W\right)_{1}\cdots\text{vec}\left(W\right)_{in\times out}b_{1}\cdots b_{out}\right)
</script>
</span>. We get the full jacobian by stacking the jacobians:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\boldsymbol{J}_{\boldsymbol{y}}^{\theta} & = & \left(\begin{array}{cc}
h^{T} & 1\end{array}\right)\otimes\boldsymbol{J}_{\boldsymbol{y}}^{a}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
From this expression we can finally express the FIM in a factorized form:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
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
</script>

</span>

</p>
<p class="Indented">
In this expression, remember that all 3 variables <span class="MathJax_Preview"><script type="math/tex">
h
</script>
</span>, <span class="MathJax_Preview"><script type="math/tex">
a
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{y}
</script>
</span> have a different value for each individual example. The FIM is the sum of the contributions of each example. The use of the Kronecker product permits splitting the contribution of each example in a term that involves the input of the layer <span class="MathJax_Preview"><script type="math/tex">
\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)
</script>
</span> and a term that involves the jacobian received on the output of the linear transformation <span class="MathJax_Preview"><script type="math/tex">
\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}D\left(\boldsymbol{y}\right)\boldsymbol{J}_{\boldsymbol{y}}^{a}
</script>
</span>.
</p>
<h1 class="Section">
KFAC
</h1>
<p class="Unindented">
<span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> use a simplification of the FIM that drastically reduce the computation required for inverting it. Remember that even if we consider a block diagonal approximation of the FIM, each block still has size <span class="MathJax_Preview"><script type="math/tex">
n_{parameters}\times n_{parameters}
</script>
</span>. We need to invert this block, and the operation of inverting a square matrix is <span class="MathJax_Preview"><script type="math/tex">
O\left(n_{parameters}^{3}\right)
</script>
</span>. They propose the following approximation:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\mathbf{F} & \approx & \mathbb{E}\left[\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)\right]\otimes\mathbb{E}\left[\left(\boldsymbol{J}_{\boldsymbol{y}}^{a}\right)^{T}D\left(\boldsymbol{y}\right)\boldsymbol{J}_{\boldsymbol{y}}^{a}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
The Kronecker product has the nice property that for 2 invertible square matrices <span class="MathJax_Preview"><script type="math/tex">
A
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
B
</script>
</span>, <span class="MathJax_Preview"><script type="math/tex">
\left(A\otimes B\right)^{-1}=A^{-1}\otimes B^{-1}
</script>
</span>. It follows that inverting the FIM now requires inverting 2 smaller matrices. However the approximation they use is questionable, and they show some interesting experimental results.
</p>
<h1 class="Section">
Natural Neural Networks
</h1>
<p class="Unindented">
<span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> exploit the same factorization <a class="Reference" href="#eq:factoredfim">↓</a> but only consider the term involving the input of the linear transformation (<span class="MathJax_Preview"><script type="math/tex">
h
</script>
</span>). They propose a reparametrization that will make <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[\left(\begin{array}{cc}
hh^{T} & h\\
h^{T} & 1
\end{array}\right)\right]
</script>
</span> equal the identity. To this view, they change the original linear transformation <span class="MathJax_Preview"><script type="math/tex">
a=Wh+b
</script>
</span> to become:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
a & = & VU\left(h-\mu\right)+d
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
<span class="MathJax_Preview"><script type="math/tex">
V
</script>
</span> is the new weight matrix and <span class="MathJax_Preview"><script type="math/tex">
d
</script>
</span> are the new biases. <span class="MathJax_Preview"><script type="math/tex">
\mu=\mathbb{E}\left[h\right]
</script>
</span> is the mean value for <span class="MathJax_Preview"><script type="math/tex">
h
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
U
</script>
</span> is the square root of the inverse covariance of <span class="MathJax_Preview"><script type="math/tex">
h
</script>
</span>, defined by <span class="MathJax_Preview"><script type="math/tex">
U^{2}=\left(\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]\right)^{-1}
</script>
</span>, denoted by <span class="MathJax_Preview"><script type="math/tex">
U=\left(\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]\right)^{-\frac{1}{2}}
</script>
</span>. <span class="MathJax_Preview"><script type="math/tex">
U
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\mu
</script>
</span> are not trained using gradient descent but instead they are estimated using data from the training set.
</p>
<p class="Indented">
Our new parameters <span class="MathJax_Preview"><script type="math/tex">
V
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
d
</script>
</span> are trained using gradient descent, which will now have the desired property. We will denote by <span class="MathJax_Preview"><script type="math/tex">
h_{e}=U\left(h-\mu\right)
</script>
</span> our new &ldquo;effective&rdquo; input to the linear transformation induced by the weight matrix <span class="MathJax_Preview"><script type="math/tex">
V
</script>
</span>. Let us first remark that <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[h_{e}\right]=U\left(\mathbb{E}\left[h\right]-\mu\right)=U\left(\mu-\mu\right)=0
</script>
</span>, so the new input is centered on average. A second remark is that <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[h_{e}h_{e}^{T}\right]=U\mathbb{E}\left[\left(h-\mu\right)\left(h-\mu\right)^{T}\right]U^{T}=\mathbf{I}
</script>
</span>. By construction <span class="MathJax_Preview"><script type="math/tex">
U
</script>
</span> cancels out the covariance. Wrapping everything together we thus have the desired property that:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
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
</script>

</span>

</p>
<p class="Indented">
The FIM for our new reparametrization thus has a better form, and they also show experimentally that this method is very efficient, and that by amortizing the cost of inverting the covariance matrix over several parameter updates, it can be faster that standard SGD.
</p>
<h1 class="Section">
Conclusion
</h1>
<p class="Unindented">
In this blog note we showed the path from the intuituion behind using the FIM as a preconditioner for gradient descent, to 2 recent algorithms that use the FIM. We also introduced the complete form for the FIM, also including the bias, which in the best of our knowledge has not been explicited in any publication at this time.
</p>
<p class="Indented">
We think that the FIM or other similar preconditioners, and their factorizations, will open a new serie of algorithms to optimize neural networks, that will help make progress in artificial intelligence tasks.
</p>
<p class="Indented">
<h1 class="biblio">
References
</h1>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-1"><span class="bib-index">1</span></a>] </span> <span class="bib-authors">Guillaume Desjardins, Karen Simonyan, Razvan Pascanu, others</span>: “<span class="bib-title">Natural neural networks</span>”, <i><span class="bib-booktitle">Advances in Neural Information Processing Systems</span></i>, pp. <span class="bib-pages">2071—2079</span>, <span class="bib-year">2015</span>.
</p>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-2"><span class="bib-index">2</span></a>] </span> <span class="bib-authors">James Martens, Roger B Grosse</span>: “<span class="bib-title">Optimizing Neural Networks with Kronecker-factored Approximate Curvature.</span>”, <i><span class="bib-booktitle">ICML</span></i>, pp. <span class="bib-pages">2408—2417</span>, <span class="bib-year">2015</span>.
</p>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-3"><span class="bib-index">3</span></a>] </span> <span class="bib-authors">Razvan Pascanu, Yoshua Bengio</span>: “<span class="bib-title">Revisiting natural gradient for deep networks</span>”, <i><span class="bib-journal">arXiv preprint arXiv:1301.3584</span></i>, <span class="bib-year">2013</span>.
</p>

</p>


<!--endhtml-->
