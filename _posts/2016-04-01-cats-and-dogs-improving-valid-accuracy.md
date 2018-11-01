---
layout: post
title: Cats and dogs - Improving the validation accuracy
subtitle: 
categories: [ift6266]
---

As I mentionned in my last post, one of the weakness of my model is that I used the same data augmentation pipeline for my validation error. The problem is that the images are rectangles, with variable width and height, and my network accepts a square image as its input. Some of the images are in landscape mode, some in portrait mode.

Without any more work, a simple way to improve the validation error is to apply the classifier to several square patches extracted from the image, then select the result with the highest confidence. Here is an example in the picture below: I extract 2 square patches (red and green) from a landscape rectangle image. 

![Patches](/img/2016-04-01-cats-and-dogs-improving-valid-accuracy/squares.png)

In the meantime in rewrote everything using the bricks provided by blocks. So we can not strictly compare the last implementation and this one, but we see that it performs similarly at the end of the training (stopped because it exceeded its walltime limit), but the curves are still decreasing. It seems we can get to a better solution by training for a couple of more hours.

You can find the learning curves below. An interesting thing is that the valid error is lower than the train error.

![Error](/img/2016-04-01-cats-and-dogs-improving-valid-accuracy/error.png)
