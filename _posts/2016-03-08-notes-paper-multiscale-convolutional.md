---
layout: post
title: Notes for paper - Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
subtitle: 
categories: [ift6268]
---

[Link to the paper](http://arxiv.org/abs/1411.4734)

## Motivation

The paper presents a technique used to solve three tasks for a given photograph:

 - what is the **depth** at any given point of the image
 - how is the surface **oriented**
 - what object contains a given pixels (**segmentation**)

The idea here is to used a single model to solve this tasks simultaneously. An interesting side effect of this is that the features used to determine the orientation and the depth are very relevant when performing segmentation. We then influence the learning in a way where each task improve all other tasks.

## Depth

As a human, looking at a photography we can tell that an object is closer to the person who holds the camera than another one.

The loss function used to measure the correctness of the depth model is defined by :

$$L_{depth} (D, D^*) = \frac 1 n \sum_i d_i^2 - \frac{1}{2 n^2} \Big( \sum_i d_i \Big)^2 + \frac 1 n \sum_i [(\nabla_x d_i)^2 + (\nabla_y d_i)^2 ]$$

where $$d_i = \log y_i - \log y_i^*$$ is the difference between the ground truth and the predicted value for pixel $$i$$ using a log scale.

The first 2 terms correspond to a scale invariant error, that is agnostic to the absolute global scale. The idea here is that we can not determine the absolute depth given an image (e.g. crop the image, how does it change the depth of every point ?), nor are we really interested in an absolute value for the depth. However, the relative depth between 2 pixels gives us information regarding whether it belongs to the same object and is much more interesting.

The last terms act as a regularizer in that it forces the predictor to be smoother but still match the ground truth.

> **Question** : why use a logarithmic distance ? It seems adapted to scenes with very distant objects such as landscapes, but is less appropriate for inner use ?

## Surface normal

Another local property of an object is the orientation of its boundary. It gives information regarding the structure of an object.

Both the ground truth and the predicted normals are normalized using $l_2$ norm. The loss then used is:

$$L_{normals} (N, N^*) = - \frac 1 n \sum_i N_i \cdot N_i^*$$

The dot product is maximized when the two vectors are aligned, which minimizes the loss.

> **Question**: This norm looks like a $$l_1$$ norm, can we think of a $$l_2$$ equivalent ? I am thinking something like $$\frac 1 n \sum_i (1 - N_i \cdot N_i^*)^2$$ that would penalize more a prediction that is far from the ground truth.

## Semantic label

The last task is semantic label. It consists in assigning an object to each pixel. An hyperparameter is the number of different object classes that we allow for a given model. the loss used here is a standard cross entropy loss:

$$L_{semantic} (C, C^*)= - \frac 1 n \sum _i C_i^* \log(C_i)$$

## Model

### Model architecture

A key point of this paper is the model used. The authors use a multiscale architecture that first predict a global estimation then refines it with higher definition network.

The idea of multi scale was first introduced in a previous paper by the same authors, the figure below show the details.

The first stage (scale 1) uses convolutions that output downscaled feature maps, which are then upsampled and fed to a second stage (scale 2) that outputs that combines them with other local features learned directly from the raw image and outputs a prediction.

![Multiscale architecture](/img/2016-03-08-notes-paper-multiscale-convolutional/multiscale.png)

This paper introduces a third stage following the same principle as in the previous paper.

The third stage outputs predictions for the three tasks.

### Training

A few tricks are used for training:

#### Training stages separately

The authors first train scales 1 and 2 first using SGD, then fix the parameters for 1 and 2 and train only scale 3.

#### Image croping

For memory reduction, during the training of scale 3, the authors use random crops as inputs and outputs instead of using the whole image. They find it speeds up training and does not impact the performance of the model.

## Conclusion

The model improves the state of the art for several datasets.

I think such a model could be used also for object recognition as it learns interesting features of a scene that can be used to recognize a certain object.