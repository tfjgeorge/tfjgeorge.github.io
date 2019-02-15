---
layout: post
title: Derivatives through a batch norm layer
comments: true
lyx: true
draft: false
categories: [note]
---
<p class="Unindented">
In this note, we will write the derivations for the backpropagated gradient through the batch norm operation, and also the gradient w.r.t the weight matrix. This derivation is very cumbersome and after having tried many different ways (column by column/line by line/element by element/etc), we present the way we think is the simplest one here.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-1">1</a> Notations
</h1>
<p class="Unindented">
We focus on a single batch normalized layer in a fully connected network. The linear part is denoted <span class="MathJax_Preview"><script type="math/tex">
y=Wx
</script>
</span> and parametrized by the weight matrix <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span>. Then the batch normalization operation is computed by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{align*}
\hat{y}=BN\left(y\right) & =\frac{y-\mu_{y}}{\sqrt{\text{var}\left(y\right)+\epsilon}}
\end{align*}
</script>

</span>

</p>
<p class="Indented">
Note that we can rewrite it in terms of <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span> directly instead of computing the intermediate step <span class="MathJax_Preview"><script type="math/tex">
y
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{align}
\hat{y} & =\frac{Wx-W\mu_{x}}{\sqrt{\text{var}\left(Wx\right)+\epsilon}}\nonumber \\
 & =\frac{W\left(x-\mu_{x}\right)}{\sqrt{\text{diag}\left(W^{\top}\text{cov}\left(x\right)W\right)+\epsilon}}\label{eq:bn_wx}
\end{align}
</script>

</span>

</p>
<p class="Indented">
Some remarks:
</p>
<ol>
<li>
These notations are not very precise since we mix up elementwise operations with linear algebra operations. Specifically, by an abuse of notation we divide a vector on the top part of the quotient, by another vector on the bottom part. 
</li>
<li>
Even if we only require to compute an elementwise variance of the components of <span class="MathJax_Preview"><script type="math/tex">
y
</script>
</span>, in <a class="Reference" href="#eq:bn_wx">\ref{eq:bn_wx}</a> we see that it hides the full covariance matrix on the vectors <span class="MathJax_Preview"><script type="math/tex">
x
</script>
</span> in minibatches, here denoted by <span class="MathJax_Preview"><script type="math/tex">
\text{cov}
</script>
</span>. It is a dense covariance matrix, with size <span class="MathJax_Preview"><script type="math/tex">
\text{in}\times\text{in}
</script>
</span>.
</li>
<li>
We did not write the scaling and bias parameters <span class="MathJax_Preview"><script type="math/tex">
\gamma
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\beta
</script>
</span> since obtaining their derivative is easier and less interesting.
</li>

</ol>
<h1 class="Section">
<a class="toc" name="toc-Section-2">2</a> Minibatch vector notation
</h1>
<p class="Unindented">
To clarify things, we consider that the examples are stacked in design matrices of size <span class="MathJax_Preview"><script type="math/tex">
\text{batch size}\times\text{vector size}
</script>
</span>:
</p>
<p class="Indented">
<div class="center">
<span class="MathJax_Preview"><script type="math/tex">
X=\left(\begin{array}{c}
-\,x^{\left(1\right)\top}\,-\\
\vdots\\
-\,x^{\left(n\right)\top}\,-
\end{array}\right)
</script>
</span>, <span class="MathJax_Preview"><script type="math/tex">
Y=\left(\begin{array}{c}
-\,y^{\left(1\right)\top}\,-\\
\vdots\\
-\,y^{\left(n\right)\top}\,-
\end{array}\right)
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\hat{Y}=\left(\begin{array}{c}
-\,\hat{y}^{\left(1\right)\top}\,-\\
\vdots\\
-\,\hat{y}^{\left(n\right)\top}\,-
\end{array}\right)
</script>
</span>
</div>

</p>
<p class="Indented">
Using this notation, we can write the result of BN for a column of the matrix (so all <span class="MathJax_Preview"><script type="math/tex">
i
</script>
</span>s component for all examples in a minibatch). We denote this column by <span class="MathJax_Preview"><script type="math/tex">
y_{i}=Y_{:i}
</script>
</span>, as opposed to the lines of <span class="MathJax_Preview"><script type="math/tex">
Y
</script>
</span> that we denoted by <span class="MathJax_Preview"><script type="math/tex">
y^{\left(j\right)\top}=Y_{j:}.
</script>
</span> Note that <span class="MathJax_Preview"><script type="math/tex">
y_{i}
</script>
</span> does not correspond to an example in the minibatch.
</p>
<p class="Indented">
We will go step by step:
</p>
<p class="Indented">
The mean of a column is obtained by multiplying it with a vector full of <span class="MathJax_Preview"><script type="math/tex">
1
</script>
</span>(denoted by a bold <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{1}
</script>
</span>), and dividing by <span class="MathJax_Preview"><script type="math/tex">
n
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{1}{n}\sum_{t}\left(y_{i}\right)_{t} & = & \frac{1}{n}\boldsymbol{1}^{\top}y_{i}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Using this we can write the (unbiased) variance of the column vector <span class="MathJax_Preview"><script type="math/tex">
y_{i}
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\text{var}\left(y_{i}\right) & = & \frac{1}{n-1}\left(y_{i}-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}y_{i}\right)^{\top}\left(y_{i}-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}y_{i}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
We multiplied the mean by a <span class="MathJax_Preview"><script type="math/tex">
\boldsymbol{1}
</script>
</span> vector in order to repeat it along all components of <span class="MathJax_Preview"><script type="math/tex">
y_{i}
</script>
</span>. We can simplify the expression:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\text{var}\left(y_{i}\right) & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{2}{n}\boldsymbol{1}\boldsymbol{1}^{\top}+\frac{1}{n^{2}}\boldsymbol{1}\boldsymbol{1}^{\top}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{2}{n}\boldsymbol{1}\boldsymbol{1}^{\top}+\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\\
 & = & \frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
And so we obtain one column of batch norm:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray}
\hat{y}_{i} & = & \frac{1}{\sqrt{\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\label{eq:bn1}
\end{eqnarray}
</script>

</span>

</p>
<p class="Indented">
Using this notation we gained the fact that everything here is linear algebra and scalar operations. We do not have any more elementwise operations, nor sums or variance, so it is easier to write derivatives using only elementary calculus rules.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-3">3</a> Jacobians and gradients
</h1>
<p class="Unindented">
Writing derivatives of vector functions with respect to vector parameters can be cumbersome, and sometimes ill-defined.
</p>
<p class="Indented">
In this note we follow the convention that a gradient of a scalar function of any object has the same shape as this object, so for instance <span class="MathJax_Preview"><script type="math/tex">
\nabla_{W}
</script>
</span> as the same shape as <span class="MathJax_Preview"><script type="math/tex">
W
</script>
</span>.
</p>
<p class="Indented">
We also make an heavy use of jacobians, which are the matrices of partial derivatives. For a function <span class="MathJax_Preview"><script type="math/tex">
f:\mathbb{R}^{m}\rightarrow\mathbb{R}^{n}
</script>
</span>, its jacobian is a <span class="MathJax_Preview"><script type="math/tex">
m\times n
</script>
</span> matrix, defined by:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\left(\frac{\partial f\left(x\right)}{\partial x}\right)_{ij} & = & \frac{\partial f\left(x\right)_{i}}{\partial x_{j}}
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Using this notation the chain rule can be written for a composition of function <span class="MathJax_Preview"><script type="math/tex">
f=g\circ h
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">

\frac{\partial f\left(x\right)}{\partial x}=\frac{\partial g\left(h\left(x\right)\right)}{\partial x}=\frac{\partial g\left(h\right)}{\partial h}\frac{\partial h\left(x\right)}{\partial x}

</script>

</span>

</p>
<p class="Indented">
Using this notation it is also easy to write first order Taylor series expansion of vector functions. The first order term is just the jacobian matrix, that we can multiply to the right by an increment <span class="MathJax_Preview"><script type="math/tex">
dx
</script>
</span>:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">

f\left(x+dx\right)=f\left(x\right)+\frac{\partial f\left(x\right)}{\partial x}dx+o\left(dx\right)

</script>

</span>

</p>
<p class="Indented">
Since <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial f\left(x\right)}{\partial x}
</script>
</span> is a <span class="MathJax_Preview"><script type="math/tex">
m\times n
</script>
</span> matrix then <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial f\left(x\right)}{\partial x}dx
</script>
</span> is a <span class="MathJax_Preview"><script type="math/tex">
m\times1
</script>
</span> column vector so it lives in the same space as <span class="MathJax_Preview"><script type="math/tex">
f
</script>
</span>. Everything works out fine!
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-4">4</a> Derivative w.r.t <span class="MathJax_Preview"><script type="math/tex">
y
</script>
</span>
</h1>
<p class="Unindented">
We start by computing the derivative through the BN operation. One of the weakness of BN is that each batch normalized feature will be a function of all other elements in a minibatch, because of the mean and variance. This is why we will focus on a single column of the design matrix <span class="MathJax_Preview"><script type="math/tex">
\hat{Y}
</script>
</span>: in this case all elements of this column only depend on the elements of the corresponding column in <span class="MathJax_Preview"><script type="math/tex">
\hat{Y}
</script>
</span>.
</p>
<p class="Indented">
We write the derivative using the expression in <a class="Reference" href="#eq:bn1">\ref{eq:bn1}</a>.
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial\hat{y}_{i}}{\partial y_{i}} & = & \frac{1}{\sqrt{\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{2}\frac{1}{\left(\frac{1}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}+\epsilon\right)^{\frac{3}{2}}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)y_{i}\frac{2}{n-1}y_{i}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
By 
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial L}{\partial y_{i}} & = & \frac{\partial L}{\partial\hat{y}_{i}}\frac{\partial\hat{y}_{i}}{\partial y_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-\frac{1}{n-1}\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{i}\hat{y}_{i}^{\top}\right)
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
Note that <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial L}{\partial y_{i}}
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial L}{\partial\hat{y}_{i}}
</script>
</span> are row vectors.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-5">5</a> Derivative w.r.t <span class="MathJax_Preview"><script type="math/tex">
Y
</script>
</span> 
</h1>
<p class="Unindented">
For efficient implementation, it is often more efficient to work with design matrices of size <span class="MathJax_Preview"><script type="math/tex">
n\times d
</script>
</span> where <span class="MathJax_Preview"><script type="math/tex">
n
</script>
</span> is the size of the minibatch, and <span class="MathJax_Preview"><script type="math/tex">
d
</script>
</span> is the feature size. With some algebraic manipulation we write the gradient for all elements in the design matrix:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
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
</script>

</span>

</p>
<p class="Indented">
we denoted by <span class="MathJax_Preview"><script type="math/tex">
\Sigma_{y}^{-1}=\left(\begin{array}{ccc}
\frac{1}{\sqrt{\text{var}\left(y_{1}\right)+\epsilon}}\\
 & \ddots\\
 &  & \frac{1}{\sqrt{\text{var}\left(y_{m}\right)+\epsilon}}
\end{array}\right)
</script>
</span> the diagonal matrix of the inverse standard deviation as usually used in BN, and <span class="MathJax_Preview"><script type="math/tex">
C=\frac{1}{n-1}\left(\begin{array}{ccc}
\frac{\partial L}{\partial\hat{y}_{i}}\hat{y}_{1} & 0 & 0\\
0 & \ddots & 0\\
0 & 0 & \frac{\partial L}{\partial\hat{y}_{m}}\hat{y}_{m}
\end{array}\right)
</script>
</span> is a diagonal matrix where the coefficients are the (scalar) covariances of the elements of <span class="MathJax_Preview"><script type="math/tex">
\frac{\partial L}{\partial\hat{y}_{i}}
</script>
</span> and <span class="MathJax_Preview"><script type="math/tex">
\hat{y}_{i}
</script>
</span>.
</p>
<h1 class="Section">
<a class="toc" name="toc-Section-6">6</a> Derivative w.r.t one line of the weight matrix
</h1>
<p class="Unindented">
Using the fact that <span class="MathJax_Preview"><script type="math/tex">
Y=XW^{\top}
</script>
</span>, we write <span class="MathJax_Preview"><script type="math/tex">
y_{i}=\left(XW^{\top}\right)_{:i}=Xw_{i}
</script>
</span>, where <span class="MathJax_Preview"><script type="math/tex">
w_{i}^{\top}=W_{i:}
</script>
</span> is a line of the weight matrix (that we transpose to obtain a column vector). We can now write the derivative using the chain rule:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial\hat{y}_{i}}{\partial w_{i}} & = & \frac{\partial\hat{y}_{i}}{\partial y_{i}}\frac{\partial y_{i}}{\partial w_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)X
\end{eqnarray*}
</script>

</span>

</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\frac{\partial L}{\partial w_{i}} & = & \frac{\partial L}{\partial\hat{y}_{i}}\frac{\partial\hat{y}_{i}}{\partial w_{i}}\\
 & = & \frac{1}{\sqrt{\text{var}\left(y_{i}\right)+\epsilon}}\frac{\partial L}{\partial\hat{y}_{i}}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}-\frac{1}{n-1}\hat{y}_{i}\hat{y}_{i}^{\top}\right)X
\end{eqnarray*}
</script>

</span>

</p>
<h1 class="Section">
<a class="toc" name="toc-Section-7">7</a> Derivative w.r.t the whole matrix
</h1>
<p class="Unindented">
Now we can stack all lines of the matrix in order to get the derivative for the whole weight matrix:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
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
</script>

</span>

</p>
<h1 class="Section">
<a class="toc" name="toc-Section-8">8</a> Derivative w.r.t the input of the batch normalized layer
</h1>
<p class="Unindented">
Using <a class="Reference" href="#eq:bn1">\ref{eq:bn1}</a> and <span class="MathJax_Preview"><script type="math/tex">
Y=XW^{\top}
</script>
</span> we can write <span class="MathJax_Preview"><script type="math/tex">
\nabla_{X}
</script>
</span> using the chain rule:
</p>
<p class="Indented">
<span class="MathJax_Preview">
<script type="math/tex;mode=display">
\begin{eqnarray*}
\nabla_{X} & = & \Sigma_{y}^{-1}\left(\nabla_{\hat{Y}}^{\top}\left(I-\frac{1}{n}\boldsymbol{1}\boldsymbol{1}^{\top}\right)-C\hat{Y}^{\top}\right)W^{\top}
\end{eqnarray*}
</script>

</span>

</p>
<h1 class="Section">
<a class="toc" name="toc-Section-9">9</a> Wrap-up and acknowledgements
</h1>
<p class="Unindented">
Now you have everything you need !
</p>
<p class="Indented">
Special thanks to César Laurent for the help and proofreading.
</p>

