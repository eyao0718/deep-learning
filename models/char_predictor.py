import numpy as np
import sys

sys.path.append("mytorch")

from gru_cell import *
from nn.linear import *


class CharacterPredictor(object):
    """CharacterPredictor class.

    This is the neural net that will run one timestep of the input
    You only need to implement the forward method of this class.
    This is to test that your GRU Cell implementation is correct when used as a GRU.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        """The network consists of a GRU Cell and a linear layer."""
        self.gru = GRUCell(input_size=input_dim, hidden_size=hidden_dim)
        self.projection = Linear(in_features=hidden_dim, out_features=num_classes)
        self.num_classes =  num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection.W = np.random.rand(num_classes, hidden_dim)

    def init_rnn_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.gru.init_weights(Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """CharacterPredictor forward.

        A pass through one time step of the input

        Input
        -----
        x: (feature_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.
        
        Returns
        -------
        logits: (num_classes)
            hidden state at current time-step.

        hnext: (hidden_dim)
            hidden state at current time-step.

        """
        hnext = self.gru.forward(x, h)
        logits = self.projection.forward(np.reshape(hnext, (1, -1)))
        return logits, hnext

def inference(net, inputs):
    """CharacterPredictor inference.

    An instance of the class defined above runs through a sequence of inputs to
    generate the logits for all the timesteps.

    Input
    -----
    net:
        An instance of CharacterPredictor.

    inputs: (seq_len, feature_dim)
            a sequence of inputs of dimensions.

    Returns
    -------
    logits: (seq_len, num_classes)
            one per time step of input..

    """
    seq_len, _ = inputs.shape
    logits = np.zeros((seq_len, net.num_classes))
    hnext = np.zeros(net.hidden_dim)
    for i in range(seq_len):
        logits[i], hnext = net.forward(inputs[i], hnext)
    return logits