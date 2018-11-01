---
layout: post
title: Cats and dogs - Notes building a simple deep learning framework
subtitle: 
categories: [ift6266]
---

For the class project, I chose not to use a framework such as Keras or Lasagne as I think building my own simplified framework is the best way to fully understand what happens under the hood.

# Minimal requirements

This framework needs to enable fast design of a theano computational graph for some common network structures:

 - multilayer perceptron
 - convolutional network
 - recurrent network

This means I will need (for the moment) the following building blocks:

 - A `convolutional` block
 - A `maxpool` block
 - An `activation` block
 - A `linear` block

The network should be able to infer the shape of its outputs and its parameters given the shape of its inputs. This makes things much easier once this is properly done, as we are then able to stack layers without even thinking of special case issues (e.g. do we need to zero-pad before a maxpool and what effect can it have on the output shape)

# Design issues

## How to implement things

Keras uses an array where all layers are sequentially appended to an array, whereas Lasagne sequentially instantiates layers where the previous layer is given as a parameter to the constructor.

For my very simple framework I chose to use only functions, that return the correct output, shape and parameters.

## Differences during training/test

For some blocks, the computational graph is slightly different. Think for instance of a Dropout block. During training it is used to randomly disable some activations thus providing some regularization.

At test time, it makes no sense adding dropout because we want to test the best model that we obtained at a certain time, which is the full model with all activations enabled.

I chose to build 2 computational graphs, one for the training, the other one for the validation. Both share the same parameters (weights, biases) as they use the same shared variables.

# Putting things together

Here is a toy example where I stack 2 `convolutional` layers followed by a relu `activation`, then a `maxpool` in the end. At each step the output of the previous layer is given as the input for the new layer.

	all_parameters = []

	#############################################
	# a first block with 2 convolutions of 32 (3, 3) filters
	output, output_test, params, output_shape = convolutional(X, X, input_shape, 32, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 32, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	
	# maxpool with size=(2, 2)
	output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

The code for this framework is in the file [layers.py](https://github.com/tfjgeorge/ift6266/blob/master/catsdogs/layers.py) on github. Feel free to use!