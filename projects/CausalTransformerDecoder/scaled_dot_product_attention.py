"""
This module computes the attention score for each query-key pair
in the input sequence using the scaled dot-product mechanism.
"""

class ScaledDotProductAttention(torch.nn.Module):
    '''
    - Calculates the dot product of queries and keys
    - Scales by the square root of the dimension
    - Applies a softmax (along last attention dimension) to generate attention weights
    - Uses dropout for regularization.

    Args:
        temperature: Scaling factor for the dot product
        attn_dropout: Dropout rate for attention weights (default: 0.1)
    
    Returns:
        (weighted sum of the values, attention weights)
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = (q @ k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = attn @ v
        return output, attn