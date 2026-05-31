import numpy as np
import sys

sys.path.append("mytorch")

from rnn_cell import *
from nn.linear import *

class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)
        self.output_size = output_size
        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size) 

        Output (y): logits

        """
        self.x = x
        self.N, T, S = x.shape
        self.T = T
        if h_0 is None:
            h_0 = np.zeros((self.num_layers, self.N, self.hidden_size))
        self.hiddens.append(h_0)

        for t in range(1, T+1):
            h_temp = np.zeros(h_0.shape)
            x_in = np.zeros((self.N, S))
            for l in range(self.num_layers):
                if l == 0:
                    x_in = self.x[:, t - 1, :]
                h_temp[l, :, :] = self.rnn[l].forward(x_in, self.hiddens[t - 1][l, :, :])
                x_in = h_temp[l, :, :]
            self.hiddens.append(h_temp)
        
        logits = self.output_layer.forward(self.hiddens[T][self.num_layers-1, :, :])
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Stores derivatives of hidden states at t for all l
        # [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        derivatives = [np.zeros((self.num_layers, self.N, self.hidden_size)) for i in range(self.T + 1)]
        dh = self.output_layer.backward(delta)
        derivatives[self.T][self.num_layers - 1, :, :] = dh
        for t in reversed(range(1, self.T + 1)):

            for l in reversed(range(self.num_layers)):

                # Get h prev l either from hiddens or x depending on the layer
                # Use dh and hiddens to get the other parameters for the backward method
                # (Recall that hiddens has an extra initial hidden state)
                h_prev_l = self.x[:, t - 1, :]
                if l > 0:
                    h_prev_l = self.hiddens[t][l - 1, :, :]
                
                dx, dh_prev_t = self.rnn[l].backward(derivatives[t][l, :, :],
                                                     self.hiddens[t][l, :, :],
                                                     h_prev_l,
                                                     self.hiddens[t - 1][l, :, :])
                
                # If you arent at the first layer, you will want to add dx to the dh from l-1th layer.
                if l > 0:
                    derivatives[t][l - 1, :, :] += dx
                
                # Update dh with the new dh from the backward pass of the rnn cell
                derivatives[t - 1][l, :, :] = dh_prev_t

        # Normalize dh by batch size since initial hidden states are also treated as
        # parameters of the network (divide by batch size)
        for t in range(self.T + 1):
            derivatives[t] /= self.N
        
        return derivatives[0]