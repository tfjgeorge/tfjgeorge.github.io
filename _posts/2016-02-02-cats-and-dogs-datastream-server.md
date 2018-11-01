---
layout: post
title: Cats and dogs - Minimum viable product
subtitle: 
categories: [ift6266]
---

I pushed a first [commit](https://github.com/tfjgeorge/ift6266/commit/37d90647b67a5dc625223debcb85adc0deb0aa6c) with a running implementation of a 1-hidden layer MLP.

The model is not very interesting (and does not learn anything relevant), but what is interesting is the separation between a data augmentation process and a train process, enabled by [Fuel](http://fuel.readthedocs.org/en/latest/).

Speaking of that, I have been struggling to get it work because of this issue : [#273](https://github.com/mila-udem/fuel/issues/273) which is quick-and-dirty fixed by setting `copy=False` in `fuel/server.py`. It fixes an error stating that numpy arrays are non aligned.