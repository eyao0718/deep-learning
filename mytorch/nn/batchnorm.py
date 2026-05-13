import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = 1 / self.N * np.sum(self.Z, axis=0, keepdims=True)
        self.V = 1 / self.N * np.sum(np.square(self.Z - self.M), axis=0, keepdims=True)

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW

        dNZdV = -1 / 2 * (self.Z - self.M) * np.power(self.V + self.eps, -3 / 2)
        dLdV = np.sum(dLdNZ * dNZdV, axis=0, keepdims=True)

        dNZdM = -np.power(self.V+self.eps,-1/2)-1/2*(self.Z-self.M)*np.power(self.V+self.eps,-3/2)*(-2/self.N*np.sum(self.Z-self.M,axis=0,keepdims=True))
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)

        dLdZ = dLdNZ * np.power(self.V+self.eps,-1/2) + dLdV * (2/self.N*(self.Z-self.M)) + 1/self.N*dLdM

        return dLdZ