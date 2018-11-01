---
layout: post
title: Cats and dogs - Some better results
subtitle: 
categories: [ift6266]
---

My last post was followed by a long time without much activity, the main reason being that I focused on a very interesting Kaggle competition that just ended. A blog post is coming.

That being said, my last implementation before that break was a network inspired by vggnet for which I get some interesting results. Here are the learning curves followed by some remarks.

![Loss](/img/2016-03-17-cats-and-dogs-some-better-results/loss.png)
![Error](/img/2016-03-17-cats-and-dogs-some-better-results/error.png)

Yeah 0% validation error ! (argument below)

# Network architecture

This network is based on the paper from the team that won ImageNet 2014 with their vggnet. It consists in stacking convolution layers with small filters and downsampling with maxpool every couple of layers. Here is the complete architecture from input to output:

 - 2 convolutions layers with 32 (3, 3) filters
 - maxpool
 - 3 convolutions layers with 64 (3, 3) filters
 - maxpool
 - 4 convolutions layers with 128 (3, 3) filters
 - maxpool
 - fully connected layer with 500 hidden units
 - sigmoid

You can find this architecture implemented [here](https://github.com/tfjgeorge/ift6266/blob/f5e5206994f0082dc3dd2536f33f6f527a0eb76b/catsdogs/models/vggnet.py)

# Some remarks

## NaNs

Similar to [Florian](https://florianbordes.wordpress.com/2016/02/16/cats-vs-dogs-2-error-rate-10/), I find that Adam optimizer is very powerful in terms of finding a good solution in a minimal number of epochs, but it also reaches a NaN value at some point. In [one of the block examples](https://github.com/mila-udem/blocks-examples/blob/master/reverse_words/__init__.py) first committed by Bart there is a trick which consists of just stopping the training as soon as the gradient contains a NaN.

I would be interested though in knowing where this NaNs come from and how to avoir getting NaNs too early in the learning.

In my case the loss curve stop being plotted when a NaN is reached, and the validation error goes to 0 because of the way it is computed.

## Sigmoid output

Some other students from the course use onehot encoded targets and use a softmax as their final layer. I instead just use a sigmoid, where a value close to 0 means that the network think the current image is a cat, and a value close to 1 for a dog.

Is there any difference in term of optimization by using this different views ? I am not sure.

Also in Yann Lecun's [efficient backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) paper some tricks are mentionned to improve the sigmoid such as choosing target value not where the sigmoid saturates but at points where the second derivative is highest. I have not seen any more litterature that mention this trick but it is worth trying.

## Valid error

My validation error is computed using the same data pipeline that involves taking a random crop of the image. This can go wrong if the random crop selected does not contain a relevant part of the image for the cats/dogs task.