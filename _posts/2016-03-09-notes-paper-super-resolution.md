---
layout: post
title: Notes for paper - Image Super-Resolution Using Deep Convolutional Networks
subtitle: 
categories: [ift6268]
---

[Link to the paper](http://arxiv.org/abs/1501.00092)

## Motivation

This paper presents a technique to address the task of improving the resolution of an image given a lower resolution version, which is called a super-resolution task. The proposed model uses a convolutional network.

## Existing techniques

Several techniques have been invented to do super-resolution:

 - **prediction models** such as linear or bicubic interpolation
 - **edge based methods**
 - **image statistical methods**
 - **example-based methods**

The latter achieved state-of-the-art performance. The basic idea is to use a dataset to learn a mapping between a low-resolution patch and its corresponding higher resolution equivalent.

## CNN for super-resolution

The author extracts a pipeline that has the desired properties for super-resolution in the general case:
 - patch extraction
 - non linear mapping
 - reconstruction

Such as pipeline is exactly what a CNN with desired structure does:

### Patch extraction and representation

The first layer of the network is a convolution of a set of filter at every patch of the image. The filters are trained using back propagation instead of explicitly building a method to select relevant filters.

### Non linear mapping

The network then maps each point in the learned representation space to another representation in the high dimension space

### Reconstruction

The reconstruction part is also done by a convolution layer, which maps the high resolution representation to the output image. This convolution learns a way of averaging each pixel accross features maps into a single pixel.

![Pipeline](/img/2016-03-09-notes-paper-super-resolution/pipeline.png)

## Experiments and conclusion

Training is done using a dataset consisting of low resolution images and their corresponding high resolution target. The loss used is MSE.

The models improves the state of the art for a dataset that they call set14.

However, their attempt at adding more layer fails, which I think is linked to the problem addressed by the residual learning strategy, that is that it is difficult for a neural network to learn an identity mapping, while this is what we are essentially doing here.

> **Question**: Why perform a bicubic interpolation first ? We could just input the lower resolution image then upsample in the network.