---
layout: post
title: The algebra of second order methods in neural networks
comments: true
lyx: true
draft: false 
categories: [note]
---

This note gives the derivations for the inverse of 2 different but related 2nd order matrices: the Fisher Information Matrix, and the Gauss-Newton approximation of the Hessian. In particular we highlight 2 centering properties that follow from the local structure of those matrices:

<ul>
<li>
we should always use a centered update for weight matrices, even if it does not follow the gradient direction (see section <a class="Reference" href="#subsec:Updating-the-weight">4.2↓</a>)
</li>
<li>
we should normalize using the (centered) covariance matrix of the activation of each layer (see section <a class="Reference" href="#subsec:KFAC-inversion">3.2↓</a>)
</li>

</ul>

Along the way, we describe the derivation of an approximate method using the properties of the Kronecker product known as KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> with corresponding parameter updates, and we give a motivation for the centering trick used in Natural Neural Networks <span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span>.

## Notations and problem statement

We denote by $$f\left(x;\theta\right)$$ the output of a fully connected neural network parametrized by its weight matrices and bias vectors grouped in a vector of paramters $$\theta$$. Let us denote by $$\ell\left(f\left(x;\theta\right),y\left(x\right)\right)$$ a loss function between the value given by the model $$f\left(x;\theta\right)$$ and the true value $$y\left(x\right)$$. $$\mathcal{L}$$ is the empirical risk on a train set of $$n$$ examples: $$\mathcal{L}\left(\theta\right)=\frac{1}{n}\sum_{i}\ell\left(f\left(x_{i};\theta\right),y\left(x_{i}\right)\right)=\frac{1}{n}\sum_{i}\ell\left(x_{i}\right)$$ where we denoted $$\ell\left(x_{i}\right)=\ell\left(f\left(x_{i};\theta\right),y\left(x_{i}\right)\right)$$ to simplify notations.


Suppose a second order update $$\theta^{t+1}=\theta^{t}-\lambda G^{-1}\nabla_{\theta}\mathcal{L}$$. $$G$$ can be the Hessian matrix or an approximation given by Gauss Newton. $$G$$ can also be the Fisher Information Matrix and in this case the update is called the natural gradient. By writing the expression for Gauss-Newton and Fisher, we observe that they share a similar structure:


$$
\begin{eqnarray*}
GN & = & \mathbb{E}_{p\left(x\right)}\left[J^{T}\frac{\partial^{2}\ell\left(x\right)}{\partial f^{2}}J\right]
\end{eqnarray*}
$$



$$
\begin{eqnarray*}
F & = & \mathbb{E}_{p\left(x\right)}\left[J^{T}D\left(x\right)J\right]
\end{eqnarray*}
$$



$$J=\frac{\partial f\left(x;\theta\right)}{\partial\theta}$$ is the jacobian matrix of the output of the network, with respect to the parameters $$\theta$$. It is of size $$n_{output}\times n_{parameters}$$. For a small change $$\Delta\theta$$ it is a first order measure of the change in the value of $$f\left(x\right)$$ or more precisely $$f\left(x;\theta+\Delta\theta\right)\approx f\left(x;\theta\right)+J\Delta\theta$$. The expression for the Fisher Information Matrix is given by <span class="bibcites">[<a class="bibliocite" name="cite-3" href="#biblio-3"><span class="bib-index">3</span></a>]</span>. Without loss of generality we denote both matrices by:$$
\begin{eqnarray*}
G & = & \mathbb{E}\left[J^{T}DJ\right]
\end{eqnarray*}
$$

## Local expression for the matrix

The matrix $$G$$ has size $$n_{parameters}\times n_{parameters}$$. For a typical neural network with several millions of parameters it is untractable to store and to invert. We usually approximate it as block diagonal, where each block is a square matrix of the size of the number of scalar parameter values for a layer. With this structure, we can invert each block separately and apply the update layer by layer: $$\theta_{l}^{t+1}=\theta_{l}^{t}-\lambda G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}$$. Let us now give an exact expression for this smaller matrix $$G_{l}$$ and its inverse $$G_{l}^{-1}$$. We call it local it the sense that it is local to a layer.

### Stacking the parameters

The computation made by a layer is given by $$h_{l+1}=f_{l}\left(W_{l}h_{l}+b_{l}\right)=f_{l}\left(a_{l}\right)$$. The parameters for this layer are a matrix $$W_{l}$$ and a vector$$b_{l}$$. But in order to write a concise expression for $$G_{l}$$ we need to stack them into a vector $$\theta_{l}$$ so that the gradient $$\nabla_{\theta_{l}}\mathcal{L}$$ is a vector and writing $$G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}$$ makes sense. To this end, we use the operator $$vec$$ that stacks the column of a matrix into a vector, i.e. for a $$2\times2$$ matrix:


$$
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
$$



Our full vector of parameters becomes:


$$
\begin{eqnarray*}
\theta_{l} & = & \left(\begin{array}{c}
vec\left(W\right)\\
b
\end{array}\right)
\end{eqnarray*}
$$

### Expressions for the jacobians<a class="Label" name="subsec:Expressions-for-the"> </a>

We now focus on the block $$G_{l}=\mathbb{E}\left[J_{l}^{T}DJ_{l}\right]$$ for layer $$l$$. We require an expression for $$J_{l}=\frac{\partial f\left(x;\theta\right)}{\partial\theta_{l}}$$. By the chain rule we separate it into a back propagated contribution $$J_{a_{l}}$$ and a local contribution:


$$
\begin{eqnarray*}
J_{l} & = & \frac{\partial f\left(x;\theta\right)}{\partial a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}\\
 & = & J_{a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}
\end{eqnarray*}
$$



To obtain an exact expression for $$\frac{\partial a_{l}}{\partial\theta_{l}}$$ we will use $$vec$$ once again with the property that $$vec\left(AXB\right)=\left(B^{T}\otimes A\right)vec\left(X\right)$$ where $$\otimes$$ is the Kronecker product:


$$
\begin{eqnarray}
a_{l} & = & W_{l}h_{l}+b_{l}\nonumber \\
 & = & vec\left(W_{l}h_{l}\right)+b_{l}\nonumber \\
 & = & vec\left(\mathbf{I}W_{l}h_{l}\right)+b_{l}\\
 & = & \left(h_{l}^{T}\otimes\mathbf{I}\right)vec\left(W_{l}\right)+b_{l}\label{eq:flattened_linear}
\end{eqnarray}
$$



In the second line we used the fact that $$W_{l}h_{l}$$ is a vector and thus $$vec\left(W_{l}h_{l}\right)=W_{l}h_{l}$$. We also introduced $$\mathbf{I}$$ the identity matrix of the same size of $$a_{l}$$. From eq <a class="Reference" href="#eq:flattened_linear">\ref{eq:flattened_linear}</a> we can directly read the jacobians:


$$
\begin{eqnarray*}
\frac{\partial a_{l}}{\partial vec\left(W_{l}\right)} & = & \left(h_{l}^{T}\otimes\mathbf{I}\right)\\
\frac{\partial a_{l}}{\partial b_{l}} & = & \mathbf{I}
\end{eqnarray*}
$$



Now using $$\theta_{l}$$:


$$
\begin{eqnarray*}
\frac{\partial a_{l}}{\partial\theta_{l}} & = & \left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)
\end{eqnarray*}
$$

### Expression for the block

Getting back to the block $$G_{l}=\mathbb{E}\left[J_{l}^{T}DJ_{l}\right]$$ we get a simple expression:


$$
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
$$



In eq <a class="Reference" href="#eq:befsim">\ref{eq:befsim}</a> we added $$1\otimes$$ as it does not change anything. Between eq <a class="Reference" href="#eq:befsim">\ref{eq:befsim}</a> and <a class="Reference" href="#eq:aftsim">\ref{eq:aftsim}</a> we used the fact that $$\left(A\otimes B\right)\left(C\otimes D\right)=AC\otimes BD$$ when $$A,B,C,D$$ have corresponding sizes (i.e. the products $$AC$$ and $$BD$$ make sense).

### Discussion

We obtained an <i>exact</i> expression for the block corresponding to layer $$l$$:


$$
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
$$



It is an expectation of Kronecker products. Note that we can not swap the expectation and the Kronecker products, and thus while the expression in eq <a class="Reference" href="#eq:exact">\ref{eq:exact}</a> is exact, the one used in KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span> is an approximation.


In eq <a class="Reference" href="#eq:exact">\ref{eq:exact}</a> we denoted by $$\left(2\right)$$ the contribution that is backpropagated, and by $$\left(1\right)$$ a contribution that is local to the parameters of the layer.

## Inverting the matrix

### KFAC drill-down

Exactly inverting this matrix can still be untractable for typical neural networks. An approximation that is easier to manipulate is proposed in KFAC <span class="bibcites">[<a class="bibliocite" name="cite-2" href="#biblio-2"><span class="bib-index">2</span></a>]</span>. The key property that we are after here is that for 2 invertible matrices $$A$$ and $$B$$ we have that $$\left(A\otimes B\right)^{-1}=A^{-1}\otimes B^{-1}$$. It becomes:


$$
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
$$



The residual $$R$$ resembles a covariance between both terms:


$$
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
$$



The conditions under which it is negligible have not been extensively studied, or at least published to the best of our knowledge. We can however remark that if one part is close to $$0$$ then the expected product will be small. This is achieved for instance if $$\left(J_{a_{l}}^{T}DJ_{a_{l}}-\mathbb{E}\left[J_{a_{l}}^{T}DJ_{a_{l}}\right]\right)$$ is small for all $$x\sim p\left(x\right)$$ the data generating distribution (recall that $$D$$ and $$J_{a_{l}}$$ depend on $$x$$). To put it into words if the value of $$J_{a_{l}}^{T}DJ_{a_{l}}$$ does not vary much for all training examples. By symmetry we can make a similar argument for $$\left(h_{l}h_{l}^{T}-\mathbb{E}\left[h_{l}h_{l}^{T}\right]\right)$$.

### KFAC inversion<a class="Label" name="subsec:KFAC-inversion"> </a>

We now have a factorized approximate expression for $$G_{l}$$:


$$
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
$$



Note that while the derivations proposed in KFAC use a single vector $$\theta$$ for all parameters of the layer $$l$$, we explicitely separated the weight matrix $$W$$ and the bias $$b$$ in section <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> which gives a slightly different expression. Thus the matrix $$A$$ is separated into 2 blocks: 2 blocks on the diagonal that correspond to the weight matrix (block $$1,1$$) and the bias (block $$2,2$$), and 2 cross-terms that explicit their interactions. 


We will see that separating the bias gives a nicer interpretation with a covariance matrix (as opposed to non-centered statistics).


We can now use the property $$\left(G_{l}^{\text{approx}}\right)^{-1}=A^{-1}\otimes B^{-1}$$. $$B^{-1}$$ can not be be further simplified, so the next part is to obtain an expression for $$A^{-1}$$:


$$
\begin{eqnarray*}
A^{-1} & = & \left(\begin{array}{cc}
\mathbb{E}\left[h_{l}h_{l}^{T}\right] & \mathbb{E}\left[h_{l}\right]\\
\mathbb{E}\left[h_{l}^{T}\right] & 1
\end{array}\right)^{-1}
\end{eqnarray*}
$$



We can use the formula for inverting a block matrix (see <a class="URL" href="https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion">Wikipedia:Block Matrix</a>). We denote by $$C=\mathbb{E}\left[h_{l}h_{l}^{T}\right]-\mathbb{E}\left[h_{l}\right]\mathbb{E}\left[h_{l}^{T}\right]$$ and we get:


$$
\begin{eqnarray*}
A^{-1} & = & \left(\begin{array}{cc}
C^{-1} & -C^{-1}\mathbb{E}\left[h_{l}\right]\\
-\mathbb{E}\left[h_{l}^{T}\right]C^{-1} & 1+\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\mathbb{E}\left[h_{l}\right]
\end{array}\right)
\end{eqnarray*}
$$



Note that $$C$$ is the covariance of $$h_{l}$$: $$C=\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]=cov\left(h_{l}\right)$$. It is centered (we substract $$\mathbb{E}\left[h_{l}\right]$$) which follows from the block matrix inversion formula, which in turns follows from the fact that we separated the bias. This motivates the use of centered statistics in second order inspired algorithms.

## Writing the update

### Derivation

Now that we have an expression for $$G_{l}^{-1}$$ we can write the product $$G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}$$ required to make an update $$\theta_{l}^{t+1}=\theta_{l}^{t}-\lambda G_{l}^{-1}\nabla_{\theta_{l}}\mathcal{L}$$. In section <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> we wrote an expression for the jacobians $$J_{l}=\frac{\partial f\left(x;\theta\right)}{\partial\theta_{l}}$$. By a similar analysis we can write the gradients $$\nabla_{\theta_{l}}\mathcal{L}$$:


$$
\begin{eqnarray*}
\nabla_{\theta_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{\theta_{l}}\ell\left(x\right)\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial\ell\left(x\right)}{\partial\theta_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial\ell\left(x\right)}{\partial a_{l}}\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\left(\frac{\partial\ell\left(x\right)}{\partial a_{l}}\right)^{T}\right]\\
 & = & \mathbb{E}\left[\left(\frac{\partial a_{l}}{\partial\theta_{l}}\right)^{T}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
$$



Using the same expressions as in <a class="Reference" href="#subsec:Expressions-for-the">2.2↑</a> we can simplify $$\frac{\partial a_{l}}{\partial\theta_{l}}=\left(\left(\begin{array}{cc}
h_{l}^{T} & 1\end{array}\right)\otimes\mathbf{I}\right)$$:


$$
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
$$



Multiplying together with $$\left(G_{l}^{\text{approx}}\right)^{-1}$$ we get the product $$\Delta_{\theta_{l}}=\left(G_{l}^{\text{approx}}\right)^{-1}\nabla_{\theta_{l}}\mathcal{L}$$:


$$
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
$$



In the first line we can read the update for $$W$$ (in fact its vectorized version $$vec\left(W\right)$$), and the second line is the update for $$b$$.

### Updating the weight matrix $$W$$<a class="Label" name="subsec:Updating-the-weight"> </a>

The new update for $$W$$ is given by:


$$
\begin{eqnarray*}
\Delta_{\text{vec}\left(W_{l}\right)} & = & C^{-1}\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
$$



We some algebraic manipulations we get back to the expression for the unflattened matrix:$$
\begin{eqnarray*}
\Delta_{\text{vec}\left(W_{l}\right)} & = & \left(C^{-1}\otimes B^{-1}\right)\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right]\\
 & = & \left(C^{-1}\otimes B^{-1}\right)\mathbb{E}\left[\text{vec}\left(\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\otimes\nabla_{a_{l}}\ell\left(x\right)\right)\right]\\
\Delta_{W_{l}} & = & B^{-1}\mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]C^{-1}
\end{eqnarray*}
$$



This is to compare with the usual gradient descent update given by:


$$
\begin{eqnarray*}
\nabla_{W_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)h_{l}^{T}\right]
\end{eqnarray*}
$$



We can notice 2 additions:

<ul>
<li>
the update is rescaled and rotated using the 2 matrices \(B^{-1}\) and \(C^{-1}\)
</li>
<li>
the expectation is centered by substracting \(\mathbb{E}\left[h_{l}\right]\)
</li>

</ul>

In addition to the derivation proposed in KFAC, by expliciting the bias we obtained 2 different centerings:

<ul>
<li>
the covariance matrix \(C\)
</li>
<li>
the expectation is centered by substracting \(\mathbb{E}\left[h_{l}\right]\)
</li>

</ul>

### Updating the bias vector $$b$$

The new update for $$b$$ is given by:


$$
\begin{eqnarray*}
\Delta_{b_{l}} & = & \mathbb{E}\left[\left(1-\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\right)B^{-1}\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
$$



This is to compare with the usual gradient descent update given by:


$$
\begin{eqnarray*}
\nabla_{b_{l}}\mathcal{L} & = & \mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\right]
\end{eqnarray*}
$$



Once again we notice that the update is scaled and rotated using $$B^{-1}$$ but there is also this strange scalar scaling $$\left(1-\mathbb{E}\left[h_{l}^{T}\right]C^{-1}\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\right)$$ for which we are not able to give an interpretation. However in practice we noted that we did not have any performance gain compared to using only $$1$$.

## Conclusion

We gave explicit derivation of the second order updates used in Gauss-Newton and in Natural gradient. By explicitly separating the weight matrices and the bias we obtained a nice centering term in both the covariance matrix used to rotate the update $$C=\mathbb{E}\left[\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]$$, and in the expectation used to compute the gradient $$\mathbb{E}\left[\nabla_{a_{l}}\ell\left(x\right)\left(h_{l}-\mathbb{E}\left[h_{l}\right]\right)^{T}\right]$$.


It is well-known that centering things is often useful. It is sometimes referred as the centering trick, or mean only batch norm. Another efficient technique called natural neural networks <span class="bibcites">[<a class="bibliocite" name="cite-1" href="#biblio-1"><span class="bib-index">1</span></a>]</span> building on the structure of the FIM mentions the trick without giving it much justification. To the best that we know this justification based on the structure of the second order matrices has not yet been contributed. We hope that this blog note can enlighten deep learning practitioners who are not very familliar with second order methods, in order to invent new approximate algorithms with more efficient updates.


## References

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-1"><span class="bib-index">1</span></a>] </span> <span class="bib-authors">Guillaume Desjardins, Karen Simonyan, Razvan Pascanu, others</span>. <span class="bib-title">Natural neural networks</span>.  <i><span class="bib-booktitle">Advances in Neural Information Processing Systems</span></i>:<span class="bib-pages">2071—2079</span>, <span class="bib-year">2015</span>.

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-2"><span class="bib-index">2</span></a>] </span> <span class="bib-authors">James Martens, Roger Grosse</span>. <span class="bib-title">Optimizing neural networks with kronecker-factored approximate curvature</span>.  <i><span class="bib-booktitle">International Conference on Machine Learning</span></i>:<span class="bib-pages">2408—2417</span>, <span class="bib-year">2015</span>.

<p class="biblio">
<span class="entry">[<a class="biblioentry" name="biblio-3"><span class="bib-index">3</span></a>] </span> <span class="bib-authors">Razvan Pascanu, Yoshua Bengio</span>. <span class="bib-title">Revisiting natural gradient for deep networks</span>. <i><span class="bib-journal">arXiv preprint arXiv:1301.3584</span></i>, <span class="bib-year">2013</span>.




