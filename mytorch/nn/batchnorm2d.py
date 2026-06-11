import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        """
        Args:
            num_features: channels
            alpha: running average hyperparameter (default 0.9)
        """

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        """inference parameters"""
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):

        if eval:
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb
            return BZ

        self.Z = Z
        self.N, self.C, self.H, self.W = Z.shape

        self.M = np.einsum('bcxy->c',
                           Z,
                           optimize='optimal') / (self.N * self.H * self.W)
        self.M = self.M[None,:,None,None]
        self.V = np.einsum('bcxy->c',
                           np.square(Z - self.M),
                           optimize='optimal') / (self.N * self.H * self.W)
        self.V = self.V[None,:,None,None]

        self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
        self.BZ = self.BW * self.NZ + self.Bb

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        return self.BZ
    
    def backward(self, dLdBZ):

        self.dLdBb = np.einsum('bcxy->c',
                               dLdBZ,
                               optimize='optimal')
        self.dLdBb = self.dLdBb[None,:,None,None]

        self.dLdBW = np.einsum('bcxy->c',
                               dLdBZ * self.NZ,
                               optimize='optimal')
        self.dLdBW = self.dLdBW[None,:,None,None]

        dLdNZ = dLdBZ * self.BW

        dNZdV = -1/2*(self.Z-self.M)*np.power(self.V+self.eps,-3/2)
        dLdV = np.einsum('bcxy->c',
                         dLdNZ*dNZdV,
                         optimize='optimal')
        dLdV = dLdV[None,:,None,None]

        dNZdM = -np.power(self.V+self.eps,-1/2)-1/2*(self.Z-self.M)*np.power(self.V+self.eps,-3/2)*(-2*(np.einsum('bcxy->c',
                                                                                                                  self.Z-self.M,
                                                                                                                  optimize='optimal')/(self.N*self.H*self.W))[None,:,None,None])
        dLdM = np.einsum('bcxy->c',
                         dLdNZ*dNZdM,
                         optimize='optimal')
        dLdM = dLdM[None,:,None,None]

        dLdZ = dLdNZ*np.power(self.V+self.eps,-1/2) + dLdV*(2/(self.N*self.H*self.W)*(self.Z-self.M)) + 1/(self.N*self.H*self.W)*dLdM

        return dLdZ