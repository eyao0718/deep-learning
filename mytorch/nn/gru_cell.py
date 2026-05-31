import numpy as np
from nn.activation import *

class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        self.r = self.r_act.forward(self.x @ self.Wrx.T + self.brx + h_prev_t @ self.Wrh.T + self.brh)
        self.z = self.z_act.forward(self.x @ self.Wzx.T + self.bzx + h_prev_t @ self.Wzh.T + self.bzh)
        self.n = self.h_act.forward(self.x @ self.Wnx.T + self.bnx + self.r * (h_prev_t @ self.Wnh.T + self.bnh))
        self.h_t = (1 - self.z) * self.n + self.z * h_prev_t
        
        return self.h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        dz = (self.hidden - self.n) * delta # H_out
        dn = (1 - self.z) * delta # H_out
        dr = self.h_act.backward(dn) * (self.hidden @ self.Wnh.T + self.bnh) # H_out

        self.dWrx = self.r_act.backward(dr)[:, None] @ self.x[None, :]
        self.dWzx = self.z_act.backward(dz)[:, None] @ self.x[None, :]
        self.dWnx = self.h_act.backward(dn)[:, None] @ self.x[None, :]

        self.dWrh = self.r_act.backward(dr)[:, None] @ self.hidden[None, :]
        self.dWzh = self.z_act.backward(dz)[:, None] @ self.hidden[None, :]
        self.dWnh = (self.h_act.backward(dn) * self.r)[:, None] @ self.hidden[None, :]

        self.dbrx = self.r_act.backward(dr)
        self.dbzx = self.z_act.backward(dz)
        self.dbnx = self.h_act.backward(dn) # H_out
        
        self.dbrh = self.r_act.backward(dr)
        self.dbzh = self.z_act.backward(dz)
        self.dbnh = self.h_act.backward(dn) * self.r

        dx = self.h_act.backward(dn)[None, :] @ self.Wnx + self.z_act.backward(dz)[None, :] @ self.Wzx + self.r_act.backward(dr)[None, :] @ self.Wrx
        dh_prev_t = delta * self.z + (self.h_act.backward(dn) * self.r)[None, :] @ self.Wnh + self.z_act.backward(dz)[None, :] @ self.Wzh + self.r_act.backward(dr)[None, :] @ self.Wrh
        
        return np.reshape(dx, -1), np.reshape(dh_prev_t, -1)