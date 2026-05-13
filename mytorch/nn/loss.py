import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)
        """

        self.A = A
        self.Y = Y
        (self.N, self.C) = A.shape

        se = (A - Y) * (A - Y)
        sse = np.ones((self.N, 1)).T @ se @ np.ones((self.C, 1))
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 / (self.N * self.C) * (self.A - self.Y)

        return dLdA

class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        (self.N, self.C) = A.shape

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
        ce = -Y * np.log(self.softmax) @ Ones_C
        sce = Ones_N.T @ ce
        L = sce / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N

        return dLdA