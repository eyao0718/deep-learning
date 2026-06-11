import numpy as np

import sys
sys.path.append('mytorch')

from Conv2d import *
from activation import *
from batchnorm2d import *


class ConvBlock(object):
	"""
	A ConvBlock consists of 2D convolution followed by 2D batch normalization
	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		self.layers = [Conv2d(in_channels,out_channels,kernel_size,stride,padding),
				 	   BatchNorm2d(num_features=out_channels)]

	def forward(self, A):
		for layer in self.layers:
			A = layer.forward(A)
		return A

	def backward(self, grad):
		for layer in reversed(self.layers):
			grad = layer.backward(grad)
		return grad

class ResBlock(object):
	def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
		self.convolution_layers =  [ConvBlock(in_channels,out_channels,filter_size,stride,padding),
							        ReLU(),
									ConvBlock(out_channels,out_channels,kernel_size=1,stride=1,padding=0)]
		self.final_activation =	ReLU()

		"""
		Check size incompatibility for addition operation b/w:
		residual_connection output & convolution_layers output
		"""
		if in_channels != out_channels or (2 * padding - filter_size) // stride + 1 != 0:
			self.residual_connection = ConvBlock(in_channels,out_channels,filter_size,stride,padding)
		else:
			self.residual_connection = Identity()

	def forward(self, A):
		Z = A
		'''Implement the forward for convolution layer'''
		for layer in self.convolution_layers:
			Z = layer.forward(Z)

		'''Add the residual connection to the output of the convolution layers'''
		Z = Z + self.residual_connection.forward(A)

		'''Pass the sum of the residual layer and convolution layer to the final activation function'''
		Z = self.final_activation.forward(Z)
		return Z

	def backward(self, grad):
		'''Implement the backward of the final activation'''
		dA = self.final_activation.backward(grad)

		'''Implement the backward of residual layer to get residual_grad'''
		residual_grad = self.residual_connection.backward(dA)

		'''Implement the backward of the convolution layer to get convlayers_grad'''
		convlayers_grad = dA
		for layer in reversed(self.convolution_layers):
			convlayers_grad = layer.backward(convlayers_grad)

		'''Add convlayers_grad and residual_grad to get the final gradient'''
		dA = residual_grad + convlayers_grad
		return dA