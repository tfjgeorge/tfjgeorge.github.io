---
layout: post
title: Cats and dogs - Some experiments and full VGGNet
subtitle: 
categories: [ift6266]
---

Since my last post, I made several new attempts at improving the test error. Here are some insights and remarks, I also report non conclusive results.

# Loss function

I experimented with squared error between the prediction and the target which performed a little bit worse for the first epochs. To give some comparison elements, I reached 70% accuracy in about 10.000 iterations with binary cross entropy (batch size=25) compared to 15.000 iterations with squared error.

# Changing target

Following advice from Pr. Bengio, I tried modifying my targets to 0.1 and 0.9 instead of 0 and 1, the idea being that forcing the target to be 1 make the network overfit faster, and we do not need it to be 100% sure of a prediction, but just more than 50% sure. This experiment was not conclusive. I did not train until convergence, but the training seemed to perform a little bit worse than with the original target values.

# Batch normalization

I used batch normalization as it seems to improve both training and regularization. It allowed to train way faster, as I reach a 85% accuracy on the validation set after 10.000 iterations (batch size=25) compared to ~35.000 iterations without BN.

# Full VGGNet

So far I had used a lighter version of the VGGNet with "only" 9 convolutional layers and a single fully connected layer with 500 units. I decided to try the full VGGNet19 which won ImageNet2014. It is composed of 16 convolutional layers (+5 pooling layers) and 3 fully connected layers.

Following the paper, I trained using momentum and batch normalization. 

Adding this layers gives me an improvement to get to a 95% accuracy on test set (Kaggle submission) which I reported on the leaderboard.

![Error](/img/2016-04-10-cats-and-dogs-full-vggnet/error.png)

# Where to go next

Now that I have a pretrained network with pretty good results, I will try to fine tune it using some common tricks:

 - decrease the learning rate
 - increase the batch size
 - use L2 regularization

The code can be viewed on my github repo: [commit](https://github.com/tfjgeorge/ift6266/commit/bac6c148c57fe56c383d997f8de76aab7b6877fa)