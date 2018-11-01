---
layout: post
title: The algebra of second order methods in neural networks
comments: true
lyx: true
draft: false 
categories: [note]
---
<p class="Unindented">
This note gives the derivations for the inverse of 2 different but related 2nd order matrices: the Fisher Information Matrix, and the Gauss-Newton approximation of the Hessian. In particular we highlight 2 centering properties that follow from the local structure of those matrices:
</p>
<ul>
<li>
we should always use a centered update for weight matrices, even if it does not follow the gradient direction (see section <a class="Reference" href="#subsec:Updating-the-weight">4.2↓</a>)
</li>
<li>
we should normalize using the (centered) covariance matrix of the activation of each layer (see section <a class="Reference" href="#subsec:KFAC-inversion">3.2↓</a>)
</li>

</ul>
<p class="Unindented">
Along the way, we describe the derivation of an approximate method using the properties of the Kronecker product known as KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> with corresponding parameter updates, and we give a motivation for the centering trick used in Natural Neural Networks <span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span>.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-1">1</a> Notations and problem statement
</h1>
<p class="Unindented">
We denote by <span class="MathJax_Preview"><script type="math/tex">
f\left(x;\theta\right)
</script>
</span> the output of a fully connected neural network parametrized by its weight matrices and bias vectors grouped in a vector of paramters <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span>. Let us denote by <span class="MathJax_Preview"><script type="math/tex">
\ell\left(f\left(x;\theta\right),y\left(x\right)\right)
</script>
</span> a loss function between the value given by the model <span class="MathJax_Preview"><script type="math/tex">
f\left(x;\theta\right)
</script>
</span> and the true value <span class="MathJax_Preview"><script type="math/tex">
y\left(x\right)
</script>
</span>. <span class="MathJax_Preview"><script type="math/tex">
\mathcal{L}
</script>
</span> is the empirical risk on a train set of <span class="MathJax_Preview"><script type="math/tex">
n
</script>
</span> examples: <span class="MathJax_Preview"><script type="math/tex">
\mathcal{L}\left(\theta\right)=\frac{1}{n}\sum_{i}\ell\left(f\left(x_{i};\theta\right),y\left(x_{i}\right)\right)=\frac{1}{n}\sum_{i}\ell\left(x_{i}\right)
</script>
</span> where we denoted <span class="MathJax_Preview"><script type="math/tex">
\ell\left(x_{i}\right)=\ell\left(f\left(x_{i};\theta\right),y\left(x_{i}\right)\right)
</script>
</span> to simplify notations.
</p>
<p class="Indented">
Suppose a second order update <span class="MathJax_Preview"><script type="math/tex">
\theta^{t+1}=\theta^{t}-\lambda G^{-1}\nabla_{\theta}\mathcal{L}
</script>
</span>. <span class="MathJax_Preview"><script type="math/tex">
G
</script>
</span> can be the Hessian matrix or an approximation given by Gauss Newton. <span class="MathJax_Preview"><script type="math/tex">
G
</script>
</span> can also be the Fisher Information Matrix and in this case the update is called the natural gradient. By writing the expression for Gauss-Newton and Fisher, we observe that they share a similar structure:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
GN & = & \mathbb{E}_{p\left(x\right)}\left[J^{T}\frac{\partial^{2}\ell\left(x\right)}{\partial f^{2}}J\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
F & = & \mathbb{E}_{p\left(x\right)}\left[J^{T}D\left(x\right)J\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
<span class="MathJax_Preview"><script type="math/tex">
J=\frac{\partial f\left(x;\theta\right)}{\partial\theta}
</script>
</span> is the jacobian matrix of the output of the network, with respect to the parameters <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span>. It is of size <span class="MathJax_Preview"><script type="math/tex">
n_{output}\times n_{parameters}
</script>
</span>. For a small change <span class="MathJax_Preview"><script type="math/tex">
\Delta\theta
</script>
</span> it is a first order measure of the change in the value of <span class="MathJax_Preview"><script type="math/tex">
f\left(x\right)
</script>
</span> or more precisely <span class="MathJax_Preview"><script type="math/tex">
f\left(x;\theta+\Delta\theta\right)\approx f\left(x;\theta\right)+J\Delta\theta
</script>
</span>. The expression for the Fisher Information Matrix is given by <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>. Without loss of generality we denote both matrices by:<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
G & = & \mathbb{E}\left[J^{T}DJ\right]
\end{eqnarray*}
</script>

</span>

</p>
<h1 class="Section">
<a class="toc" name="toc-Section-2">2</a> Local expression for the matrix
</h1>
<p class="Unindented">
The matrix <span class="MathJax_Preview"><script type="math/tex">
G
</script>
</span> has size <span class="MathJax_Preview"><script type="math/tex">
n_{parameters}\times n_{parameters}
</script>
</span>. For a typical neural network with several millions of parameters it is untractable to store and to invert. We usually approximate it as block diagonal, where each block is a square matrix of the size of the number of scalar parameter values for a layer. With this structure, we can invert each block separately and apply the update layer by layer: <span class="MathJax_Preview"><script type="math/tex">
\theta_{l}^{t+1}=\theta_{l}^{t}-\lambda G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}
</script>
</span>. Let us now give an exact expression for this smaller matrix <span class="MathJax_Preview"><script type="math/tex">
G_{l}
</script>
</span> and its inverse <span class="MathJax_Preview"><script type="math/tex">
G_{l}^{-1}
</script>
</span>. We call it local it the sense that it is local to a layer.
</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-2.1">2.1</a> Stacking the parameters
</h2>
<p class="Unindented">
The computation made by a layer is given by <span class="MathJax_Preview"><script type="math/tex">
h_{l+1}=f_{l}\left(W_{l}h_{l}+b_{l}\right)=f_{l}\left(a_{l}\right)
</script>
</span>. The parameters for this layer are a matrix <span class="MathJax_Preview"><script type="math/tex">
W_{l}
</script>
</span> and a vector<span class="MathJax_Preview"><script type="math/tex">
b_{l}
</script>
</span>. But in order to write a concise expression for <span class="MathJax_Preview"><script type="math/tex">
G_{l}
</script>
</span> we need to stack them into a vector <span class="MathJax_Preview"><script type="math/tex">
\theta_{l}
</script>
</span> so that the gradient <span class="MathJax_Preview"><script type="math/tex">
\nabla_{\theta_{l}}\mathcal{L}
</script>
</span> is a vector and writing <span class="MathJax_Preview"><script type="math/tex">
G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}
</script>
</span> makes sense. To this end, we use the operator <span class="MathJax_Preview"><script type="math/tex">
vec
</script>
</span> that stacks the column of a matrix into a vector, i.e. for a <span class="MathJax_Preview"><script type="math/tex">
2\times2
</script>
</span> matrix:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{align*}
A= & \left(\begin{array}{cc}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{array}\right) & vec\left(A\right)= & \left(\begin{array}{c}
A_{11}\\
A_{21}\\
A_{12}\\
A_{22}
\end{array}\right)
\end{align*}
</script>

</span>

</p>
<p class="Indented">
Our full vector of parameters becomes:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\theta_{l} & = & \left(\begin{array}{c}
vec\left(W\right)\\
b
\end{array}\right)
\end{eqnarray*}
</script>

</span>

</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-2.2">2.2</a> Expressions for the jacobians<a class="Label" name="subsec:Expressions-for-the"> </a>
</h2>
<p class="Unindented">
We now focus on the block <span class="MathJax_Preview"><script type="math/tex">
G_{l}=\mathbb{E}\left[J_{l}^{T}DJ_{l}\right]
</script>
</span> for layer <span class="MathJax_Preview"><script type="math/tex">
l
</script>
</span>. We require an expression for <span class="MathJax_Preview"><script type="math/tex">
J_{l}=\frac{\partial f\left(x;\theta\right)}{\partial\theta_{l}}
</script>
</span>. By the chain rule we separate it into a back propagated contribution <span class="MathJax_Preview"><script type="math/tex">
J_{a_{l}}
</script>
</span> and a local contribution:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
J_{l} & = & \frac{\partial f\left(x;\theta\right)}{\partial a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}\\
 & = & J_{a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
To obtain an exact expression for <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial a_{l}}{\partial\theta_{l}}
</script>
</span> we will use <span class="MathJax_Preview"><script type="math/tex">
vec
</script>
</span> once again with the property that <span class="MathJax_Preview"><script type="math/tex">
vec\left(AXB\right)=\left(B^{T}\otimes A\right)vec\left(X\right)
</script>
</span> where <span class="MathJax_Preview"><script type="math/tex">
\otimes
</script>
</span> is the Kronecker product:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray}
a_{l} & = & W_{l}h_{l}+b_{l}\nonumber \\
 & = & vec\left(W_{l}h_{l}\right)+b_{l}\nonumber \\
 & = & vec\left(\mathbf{I}W_{l}h_{l}\right)+b_{l}\\
 & = & \left(h_{l}^{T}\otimes\mathbf{I}\right)vec\left(W_{l}\right)+b_{l}\label{eq:flattened_linear}
\end{eqnarray}
</script>

</span>

</p>
<p class="Indented">
In the second line we used the fact that <span class="MathJax_Preview"><script type="math/tex">
W_{l}h_{l}
</script>
</span> is a vector and thus <span class="MathJax_Preview"><script type="math/tex">
vec\left(W_{l}h_{l}\right)=W_{l}h_{l}
</script>
</span>. We also introduced <span class="MathJax_Preview"><script type="math/tex">
\mathbf{I}
</script>
</span> the identity matrix of the same size of <span class="MathJax_Preview"><script type="math/tex">
a_{l}
</script>
</span>. From eq <a class="Reference" href="#eq:flattened_linear">\ref{eq:flattened_linear}</a> we can directly read the jacobians:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial a_{l}}{\partial vec\left(W_{l}\right)} & = & \left(h_{l}^{T}\otimes\mathbf{I}\right)\\
\frac{\partial a_{l}}{\partial b_{l}} & = & \mathbf{I}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Now using <span class="MathJax_Preview"><script type="math/tex">
\theta_{l}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial a_{l}}{\partial\theta_{l}} & = & \left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)
\end{eqnarray*}
</script>

</span>

</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-2.3">2.3</a> Expression for the block
</h2>
<p class="Unindented">
Getting back to the block <span class="MathJax_Preview"><script type="math/tex">
G_{l}=\mathbb{E}\left[J_{l}^{T}DJ_{l}\right]
</script>
</span> we get a simple expression:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray}
G_{l} & = & \mathbb{E}\left[\left(\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}J_{a_{l}}^{T}DJ_{a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}\right]\nonumber \\
 & = & \mathbb{E}\left[\left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)^{T}J_{a_{l}}^{T}DJ_{a_{l}}\left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)\right]\nonumber \\
 & = & \mathbb{E}\left[\left(\left(\begin{array}{c}
h_{l}\\
1
\end{array}\right)\otimes\mathbf{I}\right)\left(1\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right)\left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)\right]\label{eq:befsim}\\
 & = & \mathbb{E}\left[\left(\begin{array}{c}
h_{l}\\
1
\end{array}\right)\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right]\label{eq:aftsim}\\
 & = & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right]\nonumber 
\end{eqnarray}
</script>

</span>

</p>
<p class="Indented">
In eq <a class="Reference" href="#eq:befsim">\ref{eq:befsim}</a> we added <span class="MathJax_Preview"><script type="math/tex">
1\otimes
</script>
</span> as it does not change anything. Between eq <a class="Reference" href="#eq:befsim">\ref{eq:befsim}</a> and <a class="Reference" href="#eq:aftsim">\ref{eq:aftsim}</a> we used the fact that <span class="MathJax_Preview"><script type="math/tex">
\left(A\otimes B\right)\left(C\otimes D\right)=AC\otimes BD
</script>
</span> when <span class="MathJax_Preview"><script type="math/tex">
A,B,C,D
</script>
</span> have corresponding sizes (i.e. the products <span class="MathJax_Preview"><script type="math/tex">
AC
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
BD
</script>
</span> make sense).
</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-2.4">2.4</a> Discussion
</h2>
<p class="Unindented">
We obtained an <i>exact</i> expression for the block corresponding to layer <span class="MathJax_Preview"><script type="math/tex">
l
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray}
G_{l} & = & \mathbb{E}\left[\underbrace{\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)}_{\left(1\right)}\otimes\underbrace{J_{a_{l}}^{T}DJ_{a_{l}}}_{\left(2\right)}\right]\label{eq:exact}\\
 & = & \left(\begin{array}{cc}
\mathbb{E}\left[h_{l}h_{l}^{T}\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right] & \mathbb{E}\left[h_{l}\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right]\\
\mathbb{E}\left[h_{l}^{T}\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right] & \mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]
\end{array}\right)
\end{eqnarray}
</script>

</span>

</p>
<p class="Indented">
It is an expectation of Kronecker products. Note that we can not swap the expectation and the Kronecker products, and thus while the expression in eq <a class="Reference" href="#eq:exact">\ref{eq:exact}</a> is exact, the one used in KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> is an approximation.
</p>
<p class="Indented">
In eq <a class="Reference" href="#eq:exact">\ref{eq:exact}</a> we denoted by <span class="MathJax_Preview"><script type="math/tex">
\left(2\right)
</script>
</span> the contribution that is backpropagated, and by <span class="MathJax_Preview"><script type="math/tex">
\left(1\right)
</script>
</span> a contribution that is local to the parameters of the layer.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-3">3</a> Inverting the matrix
</h1>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-3.1">3.1</a> KFAC drill-down
</h2>
<p class="Unindented">
Exactly inverting this matrix can still be untractable for typical neural networks. An approximation that is easier to manipulate is proposed in KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span>. The key property that we are after here is that for 2 invertible matrices <span class="MathJax_Preview"><script type="math/tex">
A
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
B
</script>
</span> we have that <span class="MathJax_Preview"><script type="math/tex">
\left(A\otimes B\right)^{-1}=A^{-1}\otimes B^{-1}
</script>
</span>. It becomes:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
G_{l} & = & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\otimes J_{a_{l}}^{T}DJ_{a_{l}}\right]\\
 & = & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\right]\otimes\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]+R\\
 & \approx & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\right]\otimes\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
The residual <span class="MathJax_Preview"><script type="math/tex">
R
</script>
</span> resembles a covariance between both terms:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
R & = & \mathbb{E}\left[\left(\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)-\mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\right]\right)\otimes\left(J_{a_{l}}^{T}DJ_{a_{l}}-\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\right)\right]\\
 & = & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T}-\mathbb{E}\left[h_{l}h_{l}^{T}\right] & h_{l}-\mathbb{E}\left[h_{l}\right]\\
h_{l}^{T}-\mathbb{E}\left[h_{l}\right] & 0
\end{array}\right)\otimes\left(J_{a_{l}}^{T}DJ_{a_{l}}-\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
The conditions under which it is negligible have not been extensively studied, or at least published to the best of our knowledge. We can however remark that if one part is close to <span class="MathJax_Preview"><script type="math/tex">
0
</script>
</span> then the expected product will be small. This is achieved for instance if <span class="MathJax_Preview"><script type="math/tex">
\left(J_{a_{l}}^{T}DJ_{a_{l}}-\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\right)
</script>
</span> is small for all <span class="MathJax_Preview"><script type="math/tex">
x\sim p\left(x\right)
</script>
</span> the data generating distribution (recall that <span class="MathJax_Preview"><script type="math/tex">
D
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
J_{a_{l}}
</script>
</span> depend on <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span>). To put it into words if the value of <span class="MathJax_Preview"><script type="math/tex">
J_{a_{l}}^{T}DJ_{a_{l}}
</script>
</span> does not vary much for all training examples. By symmetry we can make a similar argument for <span class="MathJax_Preview"><script type="math/tex">
\left(h_{l}h_{l}^{T}-\mathbb{E}\left[h_{l}h_{l}^{T}\right]\right)
</script>
</span>.
</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-3.2">3.2</a> KFAC inversion<a class="Label" name="subsec:KFAC-inversion"> </a>
</h2>
<p class="Unindented">
We now have a factorized approximate expression for <span class="MathJax_Preview"><script type="math/tex">
G_{l}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
G_{l}^{\text{approx}} & = & \mathbb{E}\left[\left(\begin{array}{cc}
h_{l}h_{l}^{T} & h_{l}\\
h_{l}^{T} & 1
\end{array}\right)\right]\otimes\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\\
 & = & \left(\begin{array}{cc}
\mathbb{E}\left[h_{l}h_{l}^{T}\right] & \mathbb{E}\left[h_{l}\right]\\
\mathbb{E}\left[h_{l}^{T}\right] & 1
\end{array}\right)\otimes\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\\
 & = & A\otimes B
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Note that while the derivations proposed in KFAC use a single vector <span class="MathJax_Preview"><script type="math/tex">
\theta
</script>
</span> for all parameters of the layer <span class="MathJax_Preview"><script type="math/tex">
l
</script>
</span>, we explicitely separated the weight matrix <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> and the bias <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span> in section <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> which gives a slightly different expression. Thus the matrix <span class="MathJax_Preview"><script type="math/tex">
A
</script>
</span> is separated into 2 blocks: 2 blocks on the diagonal that correspond to the weight matrix (block <span class="MathJax_Preview"><script type="math/tex">
1,1
</script>
</span>) and the bias (block <span class="MathJax_Preview"><script type="math/tex">
2,2
</script>
</span>), and 2 cross-terms that explicit their interactions. 
</p>
<p class="Indented">
We will see that separating the bias gives a nicer interpretation with a covariance matrix (as opposed to non-centered statistics).
</p>
<p class="Indented">
We can now use the property <span class="MathJax_Preview"><script type="math/tex">
\left(G_{l}^{\text{approx}}\right)^{-1}=A^{-1}\otimes B^{-1}
</script>
</span>. <span class="MathJax_Preview"><script type="math/tex">
B^{-1}
</script>
</span> can not be be further simplified, so the next part is to obtain an expression for <span class="MathJax_Preview"><script type="math/tex">
A^{-1}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
A^{-1} & = & \left(\begin{array}{cc}
\mathbb{E}\left[h_{l}h_{l}^{T}\right] & \mathbb{E}\left[h_{l}\right]\\
\mathbb{E}\left[h_{l}^{T}\right] & 1
\end{array}\right)^{-1}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We can use the formula for inverting a block matrix (see <a class="URL" href="https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion">Wikipedia:Block Matrix</a>). We denote by <span class="MathJax_Preview"><script type="math/tex">
C=\mathbb{E}\left[h_{l}h_{l}^{T}\right]-\mathbb{E}\left[h_{l}\right]\mathbb{E}\left[h_{l}^{T}\right]
</script>
</span> and we get:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
A^{-1} & = & \left(\begin{array}{cc}
C^{-1} & -C^{-1}\mathbb{E}\left[h_{l}\right]\\
-\mathbb{E}\left[h_{l}^{T}\right]C^{-1} & 1+\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\mathbb{E}\left[h_{l}\right]
\end{array}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Note that <span class="MathJax_Preview"><script type="math/tex">
C
</script>
</span> is the covariance of <span class="MathJax_Preview"><script type="math/tex">
h_{l}
</script>
</span>: <span class="MathJax_Preview"><script type="math/tex">
C=\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]=cov\left(h_{l}\right)
</script>
</span>. It is centered (we substract <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[h_{l}\right]
</script>
</span>) which follows from the block matrix inversion formula, which in turns follows from the fact that we separated the bias. This motivates the use of centered statistics in second order inspired algorithms.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-4">4</a> Writing the update
</h1>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-4.1">4.1</a> Derivation
</h2>
<p class="Unindented">
Now that we have an expression for <span class="MathJax_Preview"><script type="math/tex">
G_{l}^{-1}
</script>
</span> we can write the product <span class="MathJax_Preview"><script type="math/tex">
G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}
</script>
</span> required to make an update <span class="MathJax_Preview"><script type="math/tex">
\theta_{l}^{t+1}=\theta_{l}^{t}-\lambda G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}
</script>
</span>. In section <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> we wrote an expression for the jacobians <span class="MathJax_Preview"><script type="math/tex">
J_{l}=\frac{\partial f\left(x;\theta\right)}{\partial\theta_{l}}
</script>
</span>. By a similar analysis we can write the gradients <span class="MathJax_Preview"><script type="math/tex">
\nabla_{\theta_{l}}\mathcal{L}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\nabla_{\theta_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{\theta_{l}}\ell\left(x\right)\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial\ell\left(x\right)}{\partial\theta_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial\ell\left(x\right)}{\partial a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\left(\frac{\partial\ell\left(x\right)}{\partial a_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Using the same expressions as in <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> we can simplify <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial a_{l}}{\partial\theta_{l}}=\left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\nabla_{\theta_{l}}\mathcal{L} & = & \mathbb{E}\left[\left(\left(\begin{array}{c}
h_{l}\\
1
\end{array}\right)\otimes\mathbf{I}\right)\nabla_{a_{l}}\ell\left(x\right)\right]\\
 & = & \mathbb{E}\left[\left(\begin{array}{c}
h_{l}\\
1
\end{array}\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Multiplying together with <span class="MathJax_Preview"><script type="math/tex">
\left(G_{l}^{\text{approx}}\right)^{-1}
</script>
</span> we get the product <span class="MathJax_Preview"><script type="math/tex">
\Delta_{\theta_{l}}=\left(G_{l}^{\text{approx}}\right)^{-1}\nabla_{\theta_{l}}\mathcal{L}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta_{\theta_{l}} & = & \left(A^{-1}\otimes B^{-1}\right)\nabla_{\theta_{l}}\mathcal{L}\\
 & = & \left(\left(\begin{array}{cc}
C^{-1} & -C^{-1}\mathbb{E}\left[h_{l}\right]\\
-\mathbb{E}\left[h_{l}^{T}\right]C^{-1} & 1+\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\mathbb{E}\left[h_{l}\right]
\end{array}\right)\otimes B^{-1}\right)\mathbb{E}\left[\left(\begin{array}{c}
h_{l}\\
1
\end{array}\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right]\\
 & = & \left(\begin{array}{c}
C^{-1}\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]\\
\mathbb{E}\left[\left(1-\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\right)B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{array}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
In the first line we can read the update for <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> (in fact its vectorized version <span class="MathJax_Preview"><script type="math/tex">
vec\left(W\right)
</script>
</span>), and the second line is the update for <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span>.
</p>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-4.2">4.2</a> Updating the weight matrix <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span><a class="Label" name="subsec:Updating-the-weight"> </a>
</h2>
<p class="Unindented">
The new update for <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> is given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta_{\text{vec}\left(W_{l}\right)} & = & C^{-1}\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We some algebraic manipulations we get back to the expression for the unflattened matrix:<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta_{\text{vec}\left(W_{l}\right)} & = & \left(C^{-1}\otimes B^{-1}\right)\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right]\\
 & = & \left(C^{-1}\otimes B^{-1}\right)\mathbb{E}\left[\text{vec}\left(\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right)\right]\\
\Delta_{W_{l}} & = & B^{-1}\mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]C^{-1}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
This is to compare with the usual gradient descent update given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\nabla_{W_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)h_{l}^{T}\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We can notice 2 additions:
</p>
<ul>
<li>
the update is rescaled and rotated using the 2 matrices <span class="MathJax_Preview"><script type="math/tex">
B^{-1}
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
C^{-1}
</script>
</span>
</li>
<li>
the expectation is centered by substracting <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[h_{l}\right]
</script>
</span>
</li>

</ul>
<p class="Unindented">
In addition to the derivation proposed in KFAC, by expliciting the bias we obtained 2 different centerings:
</p>
<ul>
<li>
the covariance matrix <span class="MathJax_Preview"><script type="math/tex">
C
</script>
</span>
</li>
<li>
the expectation is centered by substracting <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[h_{l}\right]
</script>
</span>
</li>

</ul>
<h2 class="Subsection">
<a class="toc" name="toc-Subsection-4.3">4.3</a> Updating the bias vector <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span>
</h2>
<p class="Unindented">
The new update for <span class="MathJax_Preview"><script type="math/tex">
b
</script>
</span> is given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\Delta_{b_{l}} & = & \mathbb{E}\left[\left(1-\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\right)B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
This is to compare with the usual gradient descent update given by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\nabla_{b_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Once again we notice that the update is scaled and rotated using <span class="MathJax_Preview"><script type="math/tex">
B^{-1}
</script>
</span> but there is also this strange scalar scaling <span class="MathJax_Preview"><script type="math/tex">
\left(1-\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\right)
</script>
</span> for which we are not able to give an interpretation. However in practice we noted that we did not have any performance gain compared to using only <span class="MathJax_Preview"><script type="math/tex">
1
</script>
</span>.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-5">5</a> Conclusion
</h1>
<p class="Unindented">
We gave explicit derivation of the second order updates used in Gauss-Newton and in Natural gradient. By explicitly separating the weight matrices and the bias we obtained a nice centering term in both the covariance matrix used to rotate the update <span class="MathJax_Preview"><script type="math/tex">
C=\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]
</script>
</span>, and in the expectation used to compute the gradient <span class="MathJax_Preview"><script type="math/tex">
\mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]
</script>
</span>.
</p>
<p class="Indented">
It is well-known that centering things is often useful. It is sometimes referred as the centering trick, or mean only batch norm. Another efficient technique called natural neural networks <span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> building on the structure of the FIM mentions the trick without giving it much justification. To the best that we know this justification based on the structure of the second order matrices has not yet been contributed. We hope that this blog note can enlighten deep learning practitioners who are not very familliar with second order methods, in order to invent new approximate algorithms with more efficient updates.
</p>
<p class="Indented">
<h1 class="biblio">
References
</h1>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-1"><span class="bib-index">1</span></a>] </span> <span class="bib-authors">Guillaume Desjardins, Karen Simonyan, Razvan Pascanu, others</span>. <span class="bib-title">Natural neural networks</span>.  <i><span class="bib-booktitle">Advances in Neural Information Processing Systems</span></i>:<span class="bib-pages">2071—2079</span>, <span class="bib-year">2015</span>.
</p>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-2"><span class="bib-index">2</span></a>] </span> <span class="bib-authors">James Martens, Roger Grosse</span>. <span class="bib-title">Optimizing neural networks with kronecker-factored approximate curvature</span>.  <i><span class="bib-booktitle">International Conference on Machine Learning</span></i>:<span class="bib-pages">2408—2417</span>, <span class="bib-year">2015</span>.
</p>
<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-3"><span class="bib-index">3</span></a>] </span> <span class="bib-authors">Razvan Pascanu, Yoshua Bengio</span>. <span class="bib-title">Revisiting natural gradient for deep networks</span>. <i><span class="bib-journal">arXiv preprint arXiv:1301.3584</span></i>, <span class="bib-year">2013</span>.
</p>

</p>

