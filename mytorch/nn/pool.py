import numpy as np
from resampling import *
import pdb

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        K = self.kernel
        self.A = A
        batch_size, C_in, H_in, W_in = A.shape
        H_out, W_out = H_in-K+1, W_in-K+1

        Z = np.zeros((batch_size, C_in, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                window = A[:,:,i:i+K,j:j+K]
                Z[:,:,i,j] = np.max(window, axis=(2,3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        K = self.kernel
        _, _, H_out, W_out = dLdZ.shape

        dLdA = np.zeros(self.A.shape)
        for i in range(H_out):
            for j in range(W_out):
                window = self.A[:,:,i:i+K,j:j+K]
                max_window = np.max(window, axis=(2,3), keepdims=True)
                mask = (window == max_window)
                dLdA[:,:,i:i+K,j:j+K] += (mask * dLdZ[:,:,i,j][:,:,np.newaxis,np.newaxis])

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        K = self.kernel
        batch_size, in_channels, self.input_height, self.input_width = A.shape
        output_height, output_width = self.input_height-K+1, self.input_width-K+1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                Z[:,:,i,j] = np.mean(A[:,:,i:i+K,j:j+K], axis=(2,3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        K = self.kernel
        _, in_channels, input_height, input_width = self.A.shape
        pad_dLdZ = np.pad(dLdZ, pad_width=((0,0),(0,0),(K-1,K-1),(K-1,K-1)), mode='constant')
        ft = np.ones((K,K),dtype=float) * 1.0 / (K * K)
        dLdA = np.zeros(self.A.shape)
        for i in range(input_height):
            for j in range(input_width):
                dLdA[:,:,i,j] = np.einsum('bcxy,xy->bc',
                                          pad_dLdZ[:,:,i:i+K,j:j+K],
                                          ft,
                                          optimize='optimal')
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):

        self.kernel = kernel
        self.stride = stride

        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):

        self.kernel = kernel
        self.stride = stride

        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA