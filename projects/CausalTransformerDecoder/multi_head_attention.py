"""
This module implements multi-head attention, where multiple sets of
attention heads are computed in parallel, and their outputs are concatenated.

To allow the model to jointly attend to different positions in the
input sequence from different representation subspaces.
"""

class MultiHeadAttention(torch.nn.Module):
    '''
    Multi-Head Attention Module
    - Projects the input query, key, and value matrices into multiple smaller subspaces (heads).
    - Computes scaled dot-product attention for each head in parallel.
    - Concatenates the outputs of all heads and applies a final linear transformation to project
      the result back to the original model dimension.

    Args:
        n_head: Number of attention heads
        d_model: Dimensionality of the input and output representations
        dropout: Dropout rate applied to the attention output (default: 0.1)
    
    Returns:
        (concatenated output of all attention heads, averaged attention weights)
    '''
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        # Linear layers for projecting the input query, key, and value to multiple heads
        self.w_qs   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_ks   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_vs   = torch.nn.Linear(d_model, n_head * self.d_v)

        # Initialize the weights of the linear layers
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        # Single self attention head
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5), attn_dropout=dropout)

        # Final linear layer to project the concatenated outputs of
        # the attention heads back to the model dimension
        self.fc = torch.nn.Linear(n_head * self.d_v, d_model)
        torch.nn.init.normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # following key, value, query standard computation
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # Project the input query, key, and value to multiple heads
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Rearrange the dimensions to group the heads together for parallel processing
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat the mask for each attention head if a mask is provided
        if mask is not None:
              mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1)

        # Apply scaled dot-product attention to the projected query, key, and value
        output, attn = self.attention(q, k, v, mask=mask)

        """
        This modifies the attention weighted value
        """
        # Rearrange the output back to the original order
        output = output.transpose(1, 2).contiguous()
        # Concatenate the heads
        output = output.view(sz_b, len_v, -1)

        # Final linear layer to project the concatenated outputs of
        # the attention heads back to the model dimension (d_model)
        output = self.fc(output)
        # Apply dropout
        output = self.dropout(output)

        """
        This modifies the attention weights
        """
        # average the attention weights along n_head dimension
        attn_weights = attn.mean(dim=1)

        # Returns...
        # output: (sz_b, len_v, d_model)
        # attention weights: (sz_b, len_v, len_v)
        return output, attn_weights