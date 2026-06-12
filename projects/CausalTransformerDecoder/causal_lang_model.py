"""
Causal Language Model:
    This module implements a Transformer-based decoder for
    causal language modeling (CLM). It consists of several components,
    including embedding layers, positional encoding, self-attention layers,
    and feed-forward layers. It supports various generation strategies such
    as beam search and sampling.

Key Components:
- **Embedding Layer**: Converts input tokens into dense vector representations.
- **Positional Encoding**: Adds position information to input tokens, helping the model understand the order of tokens.
- **Decoder Layers**: Composed of:
  - `DecoderLayer1`: Implements self-attention and layer normalization.
  - `DecoderLayer3`: Implements a feed-forward network with residual connections.
- **Output Linear Layer**: Projects the hidden states to the vocabulary size to generate output probabilities.

Key Methods:
- **`forward`**: Runs the input through the decoder layers and generates output probabilities.
"""




class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size=31, d_model=256, num_layers=2, num_heads=2, d_ff=512, dropout=0.1, max_length=1000):
        """
        Decoder module in the Transformer architecture.
        Initializes embeddings, multiple decoder layers, and an output linear layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): The number of expected features in the input (embedding dimension).
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward network model.
            dropout (float): Dropout probability.
            max_length (int): Maximum length of input sequences.
        """
        super(CausalLanguageModel, self).__init__()
        # convert each integer token id to an embedding of dimension d_model
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len=max_length, dropout=dropout)
        self.num_layers= num_layers
        self.dec_layers1 = nn.ModuleList([DecoderLayer1(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers3 = nn.ModuleList([DecoderLayer3(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.fully_connected = torch.nn.Linear(d_model, vocab_size)
        # recursively initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inp):
        # generate the causal mask using the given function
        attn_mask = create_mask(seq=inp, pad_idx=PAD_TOKEN)
        # convert input to embeddings
        inp = self.embedding(inp)
        # apply positional encoding
        inp = self.pos_encoder(inp)
        attention_weights_list = []

        for i in range(self.num_layers):
            # apply decoder layer
            inp, attn_weights = self.dec_layers1[i].forward(tgt=inp, attn_mask=attn_mask)
            inp = self.dec_layers3[i].forward(tgt=inp)
            attention_weights_list.append(attn_weights)

        # apply layernorm and the fully connected layer for classification
        output = self.layer_norm(inp)
        output = self.fully_connected(output)
        stacked_attention_weights = torch.stack(attention_weights_list, dim=0)
        return output, stacked_attention_weights