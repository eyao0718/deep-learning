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

class CNN_SimpleScanningMLP():

    def __init__(self):

        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):

        w1, w2, w3 = weights
        
        self.conv1.conv1d_stride1.W = w1.transpose([1,0]).reshape((self.conv1.conv1d_stride1.out_channels,
                                                                       self.conv1.conv1d_stride1.kernel_size,
                                                                       self.conv1.conv1d_stride1.in_channels
                                                                       )).transpose([0,2,1])
        
        self.conv2.conv1d_stride1.W = w2.transpose([1,0]).reshape((self.conv2.conv1d_stride1.out_channels,
                                                                       self.conv2.conv1d_stride1.kernel_size,
                                                                       self.conv2.conv1d_stride1.in_channels
                                                                       )).transpose([0,2,1])
        
        self.conv3.conv1d_stride1.W = w3.transpose([1,0]).reshape((self.conv3.conv1d_stride1.out_channels,
                                                                       self.conv3.conv1d_stride1.kernel_size,
                                                                       self.conv3.conv1d_stride1.in_channels
                                                                       )).transpose([0,2,1])
        
    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA

class CNN_DistributedScanningMLP():

    def __init__(self):

        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):

        return self.forward(A)

    def init_weights(self, weights):

        w1, w2, w3 = weights
        s_w1, s_w2, s_w3 = w1[0:48,0:2], w2[0:4,0:8], w3[:,:]

        self.conv1.conv1d_stride1.W = s_w1.transpose([1,0]).reshape((self.conv1.conv1d_stride1.out_channels,
                                                                     self.conv1.conv1d_stride1.kernel_size,
                                                                     self.conv1.conv1d_stride1.in_channels
                                                                     )).transpose([0,2,1])

        self.conv2.conv1d_stride1.W = s_w2.transpose([1,0]).reshape((self.conv2.conv1d_stride1.out_channels,
                                                                     self.conv2.conv1d_stride1.kernel_size,
                                                                     self.conv2.conv1d_stride1.in_channels
                                                                     )).transpose([0,2,1])

        self.conv3.conv1d_stride1.W = s_w3.transpose([1,0]).reshape((self.conv3.conv1d_stride1.out_channels,
                                                                     self.conv3.conv1d_stride1.kernel_size,
                                                                     self.conv3.conv1d_stride1.in_channels
                                                                     )).transpose([0,2,1])

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
    
class MLP():

    def __init__(self, layer_sizes):

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            self.layers.append(Linear(in_size, out_size))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1]

    def init_weights(self, weights):

        for i in range(len(weights)):
            self.layers[i * 2].W = weights[i].T

    def forward(self, A):

        Z = A
        for layer in self.layers:
            Z = layer(Z)
        return Z

    def backward(self, dLdZ):
        
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA