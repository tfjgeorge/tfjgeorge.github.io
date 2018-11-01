---
layout: post
title: Cats and dogs - First convolutional network trained on Calcul Québec
subtitle: 
categories: [ift6266]
---

I successfully trained a simple convolutional network composed of 2 convolutional layers. Here are some insights:

# Calcul Québec

For the course, we have been given access to a cluster called [Calcul Québec](http://www.calculquebec.ca/). With the help of this [post](https://florianbordes.wordpress.com/2016/02/09/how-to-use-the-cluster-of-calcul-quebec/) by Florian, I successfully ran a training with Fuel, Theano and Blocks. Please refer to his blog post for details.

# Convolutional network

For this first convolutional network, I could have used Keras, Lasagne or Blocks layers as a scaffold to quickly build a model. I choose not to use them as I consider interesting the use of lower level functions as we understand exactly what happens. Luckily theano already implements a convolution operator so that we do not have to define the actual convolution operation, but the (quite) tricky part is to get the right dimensions for everything: inputs, outputs, matrices and tensors.

I chose to use a fixed image size of 100x100 for input, the next paragraph describes how I get this images.

You can find the code [here](https://github.com/tfjgeorge/ift6266/tree/b0e97db6b52906f479865e6e8288f96c6ff2276e/cats%20and%20dogs).

# Data processing

As mentionned in a previous [post](/posts/2016-02-02-cats-and-dogs-datastream-server), the data pipeline uses Fuel. I [implemented](https://github.com/tfjgeorge/fuel/commit/6fe137e5ad5b21af73ddd5466e1f2ac084ac8ad0) a new Transformer that downscales images so that their biggest dimension (either width or height) is of a given size. I use it as a preprocessing brick before a RandomFixedSizeCrop. The idea is that I want to get the crop covering the largest possible area, so that even if the original image is way bigger (say 500x300), the RandomFixedSizeCrop will not select a very small part of the image that is not relevant for classification.

The full pipelines is thus :

- `MinimumImageDimensions` so that we have at least 100 pixels for each dimension
- `DownscaleMinDimension` so that the smallest dimension is 100 pixels
- `RandomFixedSizeCrop` we get a 100x100 image
- `ScaleAndShift` we transform a 8-bit integer image into a float image
- `Cast` we cast it to float32 (for use with GPU)

# Results and remarks

For this first training I did not implement early stopping, so the final network is overfitting. What is interesting is that while the loss function starts increasing for the validation set, the validation error does not. I wonder whether this is a bug in my code, or if this is expected behaviour:

I am using the binary cross entropy as my loss function:
$$f(t, o) = -(t \log(o) + (1-t) \log(1-o))$$

My guess is that as we continue overfitting, the network gets higher confidence in wrong predictions, so the log gets bigger for some examples.

![Loss](/img/2016-02-09-cats-and-dogs-first-convolutional-network-files/loss.png)
![Error](/img/2016-02-09-cats-and-dogs-first-convolutional-network-files/error.png)

# What's next

More convolutions, early stopping, visualize filters