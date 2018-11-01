---
layout: post
title: Cats and dogs - Working plan
categories: [ift6266]
---

During the winter semester, I will be working on the cats and dogs dataset which purpose is to build a machine learning system that recognizes images and classifies them as containing a cat or a dog.

This post is a draft of some ideas I am planning to implement. Some may be discarded because of lack of time or being replaced by a better one.

## Simple convolutional network

First, I will implement a simple feed forward network using convolutions as it first layers. Starting from this, I will be able to study the **influence of several hyperparameters** such as :

- number of layers
- number of filters
- size of the convolutions
- activations
- cost function

Second, since the dataset is limited in size (and not that big), we can use **data augmentation** as a way to present new examples to our training procedure, which will allow better generalization:

- crop, rotate images
- mask areas of the image
- merge images ? (of the same class obviously)
- use adversarial examples that apply worst case noise that fool the network: [Explaining and harnessing adversarial examples](http://arxiv.org/pdf/1412.6572v3.pdf)

## Next steps

### Pretraining

Unsupervised pre training for instance using **denoising auto encoders** may help to train faster, and to converge to a better solution.

### Misclassified examples

At some point it can be interesting to visualize misclassified examples and answer some questions about them:

- are they the same with different initializations but same network?
- are they the same with different models ?
- try to understand why I as a human succeed where a convolutional network fails

I hope this will inspire me for new algorithms and data augmentation procedure. If different models misclassify different images then I may be able to use some **ensemble methods** to reduce the overall misclassification error.

## Exploratory work

These are some recent ideas that where published during the last year. I am interested in getting hands-on experience for the following papers:

- **Residual learning** is a technique that make the training easier so it enables the building of much deeper networks [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)
- **Visual attention** networks [Recurrent Models of Visual Attention](http://arxiv.org/pdf/1406.6247v1.pdf)
- **Ladder networks** [Semi-Supervised Learning with Ladder Networks](http://arxiv.org/pdf/1507.02672v2.pdf)
- Can the discriminative part of a [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) be used as pretrained first layers to which we add some classification layers ?

Edit 04/02/16: Added adversarial data augmentation