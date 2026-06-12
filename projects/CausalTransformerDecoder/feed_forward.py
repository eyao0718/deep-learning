"""
The feed-forward network is a fully connected layer applied
independently to each position in the sequence after the attention layers.
"""

class FeedForward(torch.nn.Module):
    '''
    - Projection Layer (Fully Connected Layers).
    - Projects the intermediate representations to a higher-dimensional space and back to the original model dimension.
    - Consists of two linear layers with a GeLU activation function and dropout in between.

    Args:
        d_model: The input and output dimensionality of the model
        d_ff: The dimensionality of the hidden layer in the feed-forward network (default: 2048)
        dropout: Dropout rate applied after the GeLU activation (default: 0.1)

    Returns:
        The transformed input sequence passed through [Linear(), GeLU(), Dropout(), Linear()]
    '''
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first linear layer, GeLU activation, and then dropout
        x = self.dropout(F.gelu(self.linear_1(x)))
        # Apply the second linear layer to project the dimension back to d_model
        x = self.linear_2(x)
        return x