---
layout: post
title: Derivatives through a batch norm layer
comments: true
lyx: true
draft: false
categories: [note]
---

In this note, we will write the derivations for the backpropagated gradient through the batch norm operation, and also the gradient w.r.t the weight matrix. This derivation is very cumbersome and after having tried many different ways (column by column/line by line/element by element/etc), we present the way we think is the simplest one here.

## Notations

We focus on a single batch normalized layer in a fully connected network. The linear part is denoted $$y=Wx$$ and parametrized by the weight matrix $$W$$. Then the batch normalization operation is computed by:

$$
\begin{align*}
\hat{y}=BN\left(y\right) & =\frac{y-\mu_{y}}{\sqrt{\text{var}\left(y\right)+\epsilon}}
\end{align*}
$$


Note that we can rewrite it in terms of $$x$$ and $$W$$ directly instead of computing the intermediate step $$y$$:

$$
\begin{align}
\hat{y} & =\frac{Wx-W\mu_{x}}{\sqrt{\text{var}\left(Wx\right)+\epsilon}}\nonumber \\
 & =\frac{W\left(x-\mu_{x}\right)}{\sqrt{\text{diag}\left(W^{\top}\text{cov}\left(x\right)W\right)+\epsilon}}\label{eq:bn_wx}
\end{align}
$$


Some remarks:

<ol>
<li>These notations are not very precise since we mix up elementwise operations with linear algebra operations. Specifically, by an abuse of notation we divide a vector on the top part of the quotient, by another vector on the bottom part. </li>
<li>Even if we only require to compute an elementwise variance of the components of \(y\), in <a class="Reference" href="#eq:bn_wx">\(\ref{eq:bn_wx}\)</a> we see that it hides the full covariance matrix on the vectors \(x\) in minibatches, here denoted by \(\text{cov}\). It is a dense covariance matrix, with size \(\text{in}\times\text{in}\).</li>
<li>We did not write the scaling and bias parameters \(\gamma\) and \(\beta\) since obtaining their derivative is easier and less interesting.</li>
</ol>

## Minibatch vector notation

To clarify things, we consider that the examples are stacked in design matrices of size $$\text{batch size}\times\text{vector size}$$:

$$
X=\left(\begin{array}{c}
-\,x^{\left(1\right)\top}\,-\\
\vdots\\
-\,x^{\left(n\right)\top}\,-
\end{array}\right)$$, $$Y=\left(\begin{array}{c}
-\,y^{\left(1\right)\top}\,-\\
\vdots\\
-\,y^{\left(n\right)\top}\,-
\end{array}\right)$$ and $$\hat{Y}=\left(\begin{array}{c}
-\,\hat{y}^{\left(1\right)\top}\,-\\
\vdots\\
-\,\hat{y}^{\left(n\right)\top}\,-
\end{array}\right)
$$


Using this notation, we can write the result of BN for a column of the matrix (so all $$i$$s component for all examples in a minibatch). We denote this column by $$y_{i}=Y_{:i}$$, as opposed to the lines of $$Y$$ that we denoted by $$y^{\left(j\right)\top}=Y_{j:}.$$ Note that $$y_{i}$$ does not correspond to an example in the minibatch.

We will go step by step:

The mean of a column is obtained by multiplying it with a vector full of $$1$$(denoted by a bold $$\boldsymbol{1}$$), and dividing by $$n$$:

$$
\begin{eqnarray*}
\frac{1}{n}\sum_{t}\left(y_{i}\right)_{t} & = & \frac{1}{n}\boldsymbol{1}^{\top}y_{i}
\end{eqnarray*}
$$


Using this we can write the (unbiased) variance of the column vector $$y_{i}$$:

$$
\begin{eqnarray*}
\text{var}\left(y_{i}\right) & = & \frac{1}{n-1}\left(y_{i}-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}y_{i}\right)^{\top}\left(y_{i}-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}y_{i}\right)
\end{eqnarray*}
$$


We multiplied the mean by a $$\boldsymbol{1}$$ vector in order to repeat it along all components of $$y_{i}$$. We can simplify the expression:

$$
\begin{eqnarray*}
\text{var}\left(y_{i}\right) & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{2}{n}\boldsymbol{1}\boldsymbol{1}^{\top}+\frac{1}{n^{2}}\boldsymbol{1}\boldsymbol{1}^{\top}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{2}{n}\boldsymbol{1}\boldsymbol{1}^{\top}+\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}
\end{eqnarray*}
$$


And so we obtain one column of batch norm:

$$
\begin{eqnarray}
\hat{y}_{i} & = & \frac{1}{\sqrt{\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\label{eq:bn1}
\end{eqnarray}
$$


Using this notation we gained the fact that everything here is linear algebra and scalar operations. We do not have any more elementwise operations, nor sums or variance, so it is easier to write derivatives using only elementary calculus rules.

## Jacobians and gradients

Writing derivatives of vector functions with respect to vector parameters can be cumbersome, and sometimes ill-defined.

In this note we follow the convention that a gradient of a scalar function of any object has the same shape as this object, so for instance $$\nabla_{W}$$ as the same shape as $$W$$.

We also make an heavy use of jacobians, which are the matrices of partial derivatives. For a function $$f:\mathbb{R}^{m}\rightarrow\mathbb{R}^{n}$$, its jacobian is a $$m\times n$$ matrix, defined by:

$$
\begin{eqnarray*}
\left(\frac{\partial f\left(x\right)}{\partial x}\right)_{ij} & = & \frac{\partial f\left(x\right)_{i}}{\partial x_{j}}
\end{eqnarray*}
$$

Using this notation the chain rule can be written for a composition of function $$f=g\circ h$$:

$$
\frac{\partial f\left(x\right)}{\partial x}=\frac{\partial g\left(h\left(x\right)\right)}{\partial x}=\frac{\partial g\left(h\right)}{\partial h}\frac{\partial h\left(x\right)}{\partial x}
$$


Using this notation it is also easy to write first order Taylor series expansion of vector functions. The first order term is just the jacobian matrix, that we can multiply to the right by an increment $$dx$$:

$$
f\left(x+dx\right)=f\left(x\right)+\frac{\partial f\left(x\right)}{\partial x}dx+o\left(dx\right)
$$

Since $$\frac{\partial f\left(x\right)}{\partial x}$$ is a $$m\times n$$ matrix then $$\frac{\partial f\left(x\right)}{\partial x}dx$$ is a $$m\times1$$ column vector so it lives in the same space as $$f$$. Everything works out fine!

## Derivative w.r.t $$y$$

We start by computing the derivative through the BN operation. One of the weakness of BN is that each batch normalized feature will be a function of all other elements in a minibatch, because of the mean and variance. This is why we will focus on a single column of the design matrix $$\hat{Y}$$: in this case all elements of this column only depend on the elements of the corresponding column in $$\hat{Y}$$.

We write the derivative using the expression in <a class="Reference" href="#eq:bn1">\ref{eq:bn1}</a>.

$$
\begin{eqnarray*}
\frac{\partial\hat{y}_{i}}{\partial y_{i}} & = & \frac{1}{\sqrt{\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{2}\frac{1}{\left(\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon\right)^{\frac{3}{2}}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\frac{2}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)
\end{eqnarray*}
$$


By 

$$
\begin{eqnarray*}
\frac{\partial L}{\partial y_{i}} & = & \frac{\partial L}{\partial\hat{y}_{i}}\frac{\partial\hat{y}_{i}}{\partial y_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{n-1}\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{i}\hat{y}_{i}^{\top}\right)
\end{eqnarray*}
$$


Note that $$\frac{\partial L}{\partial y_{i}}$$ and $$\frac{\partial L}{\partial\hat{y}_{i}}$$ are row vectors.

## Derivative w.r.t $$Y$$

For efficient implementation, it is often more efficient to work with design matrices of size $$n\times d$$ where $$n$$ is the size of the minibatch, and $$d$$ is the feature size. With some algebraic manipulation we write the gradient for all elements in the design matrix:

$$
\begin{eqnarray*}
\nabla_{Y} & = & \left(\begin{array}{ccc}
| &  & |\\
\left(\frac{\partial L}{\partial y_{1}}\right)^{\top} &  & \left(\frac{\partial L}{\partial y_{n}}\right)^{\top}\\
| &  & |
\end{array}\right)\\
 & = & \left(\begin{array}{c}
-\,\frac{\partial L}{\partial y_{1}}\,-\\
\vdots\\
-\,\frac{\partial L}{\partial y_{n}}\,-
\end{array}\right)^{\top}\\
 & = & \left(\begin{array}{c}
\frac{1}{\sqrt{\text{var}\left(y_{1}\right)+\epsilon}}\left(\frac{\partial L}{\partial\hat{y}_{1}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{n-1}\frac{\partial L}{\partial\hat{y}_{1}}\hat{y}_{1}\hat{y}_{1}^{\top}\right)\\
\vdots\\
\frac{1}{\sqrt{\text{var}\left(y_{n}\right)+\epsilon}}\left(\frac{\partial L}{\partial\hat{y}_{n}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{n-1}\frac{\partial L}{\partial\hat{y}_{n}}\hat{y}_{n}\hat{y}_{n}^{\top}\right)
\end{array}\right)^{\top}\\
 & = & \left(\left(\begin{array}{c}
-\,\frac{\partial L}{\partial\hat{y}_{i}}\,-\\
\vdots\\
-\,\frac{\partial L}{\partial\hat{y}_{i}}\,-
\end{array}\right)\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{n-1}\left(\begin{array}{ccc}
\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{1} & 0 & 0\\
0 & \ddots & 0\\
0 & 0 & \frac{\partial L}{\partial\hat{y}_{m}}\hat{y}_{m}
\end{array}\right)\left(\begin{array}{c}
-\,\hat{y}_{1}^{\top}\,-\\
\vdots\\
-\,\hat{y}_{m}^{\top}\,-
\end{array}\right)\right)^{\top}\Sigma_{y}^{-1}\\
 & = & \left(\nabla_{\hat{Y}}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-C\hat{Y}^{\top}\right)^{\top}\Sigma_{y}^{-1}\\
 & = & \left(\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)\nabla_{\hat{Y}}-\hat{Y}C\right)^{\top}\Sigma_{y}^{-1}
\end{eqnarray*}
$$


we denoted by $$\Sigma_{y}^{-1}=\left(\begin{array}{ccc}
\frac{1}{\sqrt{\text{var}\left(y_{1}\right)+\epsilon}}\\
 & \ddots\\
 &  & \frac{1}{\sqrt{\text{var}\left(y_{m}\right)+\epsilon}}
\end{array}\right)$$ the diagonal matrix of the inverse standard deviation as usually used in BN, and $$C=\frac{1}{n-1}\left(\begin{array}{ccc}
\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{1} & 0 & 0\\
0 & \ddots & 0\\
0 & 0 & \frac{\partial L}{\partial\hat{y}_{m}}\hat{y}_{m}
\end{array}\right)$$ is a diagonal matrix where the coefficients are the (scalar) covariances of the elements of $$\frac{\partial L}{\partial\hat{y}_{i}}$$ and $$\hat{y}_{i}$$.

## Derivative w.r.t one line of the weight matrix

Using the fact that $$Y=XW^{\top}$$, we write $$y_{i}=\left(XW^{\top}\right)_{:i}=Xw_{i}$$, where $$w_{i}^{\top}=W_{i:}$$ is a line of the weight matrix (that we transpose to obtain a column vector). We can now write the derivative using the chain rule:

$$
\begin{eqnarray*}
\frac{\partial\hat{y}_{i}}{\partial w_{i}} & = & \frac{\partial\hat{y}_{i}}{\partial y_{i}}\frac{\partial y_{i}}{\partial w_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)X
\end{eqnarray*}
$$


$$
\begin{eqnarray*}
\frac{\partial L}{\partial w_{i}} & = & \frac{\partial L}{\partial\hat{y}_{i}}\frac{\partial\hat{y}_{i}}{\partial w_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)X
\end{eqnarray*}
$$

## Derivative w.r.t the whole matrix

Now we can stack all lines of the matrix in order to get the derivative for the whole weight matrix:

$$
\begin{eqnarray*}
\nabla_{W} & = & \Sigma_{y}^{-1}\left(\begin{array}{c}
\frac{\partial L}{\partial\hat{y}_{1}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{1}\hat{y}_{1}^{\top}\right)X\\
\vdots\\
\frac{\partial L}{\partial\hat{y}_{m}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{m}\hat{y}_{m}^{\top}\right)X
\end{array}\right)\\
 & = & \Sigma_{y}^{-1}\left(\left(\begin{array}{c}
\frac{\partial L}{\partial\hat{y}_{i}}\\
\vdots\\
\frac{\partial L}{\partial\hat{y}_{i}}
\end{array}\right)\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)X-\frac{1}{n-1}\left(\begin{array}{ccc}
\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{1} & 0 & 0\\
0 & \ddots & 0\\
0 & 0 & \frac{\partial L}{\partial\hat{y}_{m}}\hat{y}_{m}
\end{array}\right)\left(\begin{array}{c}
\hat{y}_{1}^{\top}\\
\vdots\\
\hat{y}_{m}^{\top}
\end{array}\right)X\right)\\
 & = & \Sigma_{y}^{-1}\left(\nabla_{\hat{Y}}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-C\hat{Y}^{\top}\right)X
\end{eqnarray*}
$$

## Derivative w.r.t the input of the batch normalized layer

Using <a class="Reference" href="#eq:bn1">\ref{eq:bn1}</a> and $$Y=XW^{\top}$$ we can write $$\nabla_{X}$$ using the chain rule:

$$
\begin{eqnarray*}
\nabla_{X} & = & \Sigma_{y}^{-1}\left(\nabla_{\hat{Y}}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-C\hat{Y}^{\top}\right)W^{\top}
\end{eqnarray*}
$$

## Wrap-up and acknowledgements

Now you have everything you need !

Special thanks to César Laurent for the help and proofreading.


