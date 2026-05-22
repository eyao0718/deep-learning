import numpy as np
from resampling import *
import pdb


class Conv1d_stride1():

    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        B, _, W_in = A.shape
        W_out = W_in-self.kernel_size+1
        self.A = A

        Z = np.zeros((B, self.out_channels, W_out))
        for i in range(W_out):
            Z[:,:,i] = np.einsum('bck,ock->bo',
                                 A[:,:,i:i+self.kernel_size],
                                 self.W,
                                 optimize='optimal')
        Z = Z + self.b[np.newaxis,:,np.newaxis]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        self.dLdb = np.einsum('bcs->c', dLdZ, optimize='optimal')

        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))
        for i in range(self.kernel_size):
            self.dLdW[:,:,i] = np.einsum('bis,bos->oi',
                                         self.A[:,:,i:i+dLdZ.shape[2]],
                                         dLdZ,
                                         optimize='optimal')

        pad_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1, self.kernel_size-1)), mode='constant')
        flip_W = np.flip(self.W, axis=2)
        dLdA = np.zeros(self.A.shape)
        for i in range(self.A.shape[2]):
            dLdA[:,:,i] = np.einsum('bos,ois->bi',
                                    pad_dLdZ[:,:,i:i+self.kernel_size],
                                    flip_W,
                                    optimize='optimal')
        return dLdA

class Conv1d():

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.padding = padding

        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels,
                                             kernel_size,
                                             weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        pad_A = np.pad(A, ((0,0),(0,0),(self.padding,self.padding)), mode='constant')

        Z = self.conv1d_stride1.forward(pad_A)
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dLdA = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdA)

        dLdA = dLdA[:,:,self.padding:dLdA.shape[2]-self.padding]

        return dLdA