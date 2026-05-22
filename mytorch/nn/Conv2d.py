import numpy as np
from resampling import *


class Conv2d_stride1():

    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels,
                                               kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels,
                                    kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        K = self.kernel_size
        batch_size, _, input_height, input_width = A.shape
        output_height, output_width = input_height - K + 1, input_width - K + 1
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                Z[:,:,i,j] = np.einsum('bixy,oixy->bo',
                                       A[:,:,i:i+K,j:j+K],
                                       self.W,
                                       optimize='optimal')
        Z = Z + self.b[np.newaxis,:,np.newaxis,np.newaxis]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        K = self.kernel_size
        input_height, input_width = self.A.shape[2], self.A.shape[3]
        output_height, output_width = input_height - K + 1, input_width - K + 1

        self.dLdW = np.zeros(self.W.shape)
        for i in range(K):
            for j in range(K):
                self.dLdW[:,:,i,j] = np.einsum('bixy,boxy->oi',
                                               self.A[:,:,i:i+output_height,j:j+output_width],
                                               dLdZ,
                                               optimize='optimal')

        self.dLdb = np.einsum('boxy->o', dLdZ, optimize='optimal')

        pad_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(K-1,K-1),(K-1,K-1)), mode='constant')
        flip_W = np.flip(self.W, axis=(2,3))
        dLdA = np.zeros(self.A.shape)
        for i in range(input_height):
            for j in range(input_width):
                dLdA[:,:,i,j] = np.einsum('boxy,oixy->bi',
                                          pad_dLdZ[:,:,i:i+K,j:j+K],
                                          flip_W,
                                          optimize='optimal')
        
        return dLdA

class Conv2d():

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        
        self.stride = stride
        self.padding = padding

        self.conv2d_stride1 = Conv2d_stride1(in_channels,
                                             out_channels,
                                             kernel_size,
                                             weight_init_fn,
                                             bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        P = self.padding
        pad_A = np.pad(A, ((0,0),(0,0),(P,P),(P,P)), mode='constant')

        Z = self.conv2d_stride1.forward(pad_A)
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
        dLdA = self.conv2d_stride1.backward(dLdA)

        P = self.padding
        dLdA = dLdA[:,:,P:dLdA.shape[2]-P,P:dLdA.shape[3]-P]

        return dLdA