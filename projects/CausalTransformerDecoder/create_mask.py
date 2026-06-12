def create_mask(seq, pad_idx=None):
    """
    Create a mask to prevent positions from attending to subsequent positions.

    Args:
        seq: The input sequence tensor, shape (batch_size, sequence_length).
        pad_idx: (Optional) Padding index for masking padding positions.

    Returns:
        mask: A mask tensor with shape (batch_size, sequence_length, sequence_length),
        where the upper triangular portion is filled with 1s to prevent attention to future positions.
        The positions are allowed to attend to previous positions but not to subsequent positions.
    """
    batch_size, seq_length = seq.size()

    # Create an upper triangular matrix
    # 0 <= diagonal -> allowed positions = False
    # 1  > diagonal -> disallowed positions = True
    mask = torch.ones(seq_length, seq_length).triu(diagonal=1).to(device=DEVICE)

    # Expand the mask to match the dimensions of the key-query attention matrix
    # batch_size x seq_length x seq_length
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    if pad_idx != None:
        # Create a mask where padding positions in the key sequence are marked to true = disallowed
        pad_mask = seq.eq(pad_idx).to(device=DEVICE)
        # Expand the mask to match the dimensions of the key-query attention matrix
        # batch_size x seq_length x seq_length
        pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_length, -1)

        # equivalent to boolean OR
        mask = mask + pad_mask
    
    return mask.gt(0)

"""Uncomment to plot mask"""
# causal_mask = create_mask(seq=torch.randn((4, 10)), pad_idx=PAD_TOKEN).to("cpu")
# # Black portions = allowed
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# axs.imshow(causal_mask[0], cmap="gray", aspect='auto')
# axs.set_title("Decoder Causal Self-Attn Mask")
# plt.show()