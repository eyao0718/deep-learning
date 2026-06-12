"""
Transformers do not inherently capture the order of sequences,
so positional encodings are used to introduce sequence order into the model.
"""

class PositionalEncoding(torch.nn.Module):
    """
    Adds information about the position of each token in the input sequence.
    Uses a combination of sine and cosine functions of different frequencies to generate positional encodings.

    Args:
        projection_size: dimension D of input embedding, aka size of the input embeddings (i.e., d_model)
        max_seq_len: max T, aka maximum length of the input sequence (default: 1000)
    
    Returns:
        input embedding enriched with positional information,
        which is passed through a dropout layer for regularization
    """
    def __init__(self, projection_size, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, projection_size)
        t = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) # max_seq_len * 1
        w = torch.exp(torch.arange(0, projection_size, 2).float() * (-1.0 * torch.log(torch.tensor([10000.0])).item() / projection_size))
        pe[:, 0::2] = torch.sin(w * t)
        pe[:, 1::2] = torch.cos(w * t)
        pe = pe.unsqueeze(0) # account for batch dim broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])