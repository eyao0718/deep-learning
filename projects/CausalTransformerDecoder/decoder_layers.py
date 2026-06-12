"""
Transformer Decoder Layers

The `DecoderLayer1` and `DecoderLayer3` are modular components of the Transformer decoder.
Each layer is designed to handle a specific function: self-attention, cross-attention, and feed-forward processing.
"""



"""
DecoderLayer1: Self-Attention Layer
- **Purpose**: Implements self-attention, where the decoder attends to its own inputs,
               combined with residual connections and layer normalization.
- **Components**:
  - `MultiHeadAttention`: Applies self-attention to the target sequence.
  - `LayerNorm`: Normalizes the output after the residual connection.
  - `Dropout`: Regularization to prevent overfitting.
"""
class DecoderLayer1(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        DecoderLayer (attention and layer norm) in the Transformer architecture.

        Args:
            d_model (int): The number of expected features in the input (embedding dimension).
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward network model.
            dropout (float): Dropout probability.
        """
        super(DecoderLayer1, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.d_ff = d_ff
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None):
        # layer norm to input
        tgt_norm = self.layer_norm(tgt)

        # call self attention with mask
        attn_output, attn_weights = self.self_attn(tgt_norm, tgt_norm, tgt_norm, mask=attn_mask)

        # apply dropout
        attn_output = self.dropout(attn_output)

        # add skip connection
        tgt = tgt + attn_output

        return tgt, attn_weights



"""
DecoderLayer3: Feed-Forward Layer
- **Purpose**: Implements a feed-forward neural network for further transformation
               of the decoder's intermediate representations.
- **Components**:
  - `FeedForward`: A two-layer fully connected network with non-linearity.
  - `LayerNorm`: Applied after the residual connection.
  - `Dropout`: Regularization to avoid overfitting.
"""
class DecoderLayer3(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Feedforward layer with layer normalization in the Transformer decoder.

        Args:
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
        """
        super(DecoderLayer3, self).__init__()
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, tgt):
        # layer norm to input
        tgt_norm = self.layer_norm(tgt)

        # call feed forward layer
        ffn_output = self.ffn(tgt_norm)

        # apply dropout
        ffn_output = self.dropout(ffn_output)

        # add skip connection
        tgt = tgt + ffn_output
        
        return tgt