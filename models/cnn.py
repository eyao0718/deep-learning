from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *

import numpy as np
import os
import sys
import pdb

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr

        self.convolutional_layers = [Conv1d(num_input_channels,
                                            num_channels[0],
                                            kernel_sizes[0],
                                            strides[0],
                                            weight_init_fn=conv_weight_init_fn,
                                            bias_init_fn=bias_init_fn)
                                            ]
        
        for i in range(1, self.nlayers):
            self.convolutional_layers.append(Conv1d(in_channels=num_channels[i-1],
                                                    out_channels=num_channels[i],
                                                    kernel_size=kernel_sizes[i],
                                                    stride=strides[i],
                                                    weight_init_fn=conv_weight_init_fn,
                                                    bias_init_fn=bias_init_fn))
    
        self.flatten = Flatten()
        
        linear_input = input_width
        for i in range(self.nlayers):
            linear_input = (linear_input - kernel_sizes[i]) // strides[i] + 1
        
        self.linear_layer = Linear(in_features=linear_input*num_channels[self.nlayers-1], out_features=num_linear_neurons)

        self.layers = []
        
        for i in range(self.nlayers):
            self.layers.extend([self.convolutional_layers[i],
                                self.activations[i]])
        
        self.layers.extend([self.flatten,
                            self.linear_layer])
        
        out_features, in_features = self.linear_layer.W.shape
        if linear_weight_init_fn is not None:
            self.linear_layer.W = linear_weight_init_fn(out_features, in_features)
        if bias_init_fn is not None:
            self.linear_layer.b = bias_init_fn(out_features)

        self.L = len(self.layers)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """
        for i in range(self.L):
            A = self.layers[i].forward(A)
        self.Z = A
        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        for i in reversed(range(self.L)):
            grad = self.layers[i].backward(grad)

        return grad

    def zero_grads(self):

        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):

        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):

        self.train_mode = True

    def eval(self):

        self.train_mode = False