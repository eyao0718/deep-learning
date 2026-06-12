config = {
    # TODO: select a tokenzier from ["char", "1k", "10k"]
    "token_type": "1k",
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_length": 1000,
    "lr": 1e-3,
    "batch_size": 128,
    "num_epochs": 50,
}