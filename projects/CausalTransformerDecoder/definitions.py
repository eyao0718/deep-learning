"""
Model, Loss, Optimizer, and Scheduler Definitions
"""

# Define the model
model = CausalLanguageModel(
    vocab_size = VOCAB_SIZE,
    d_model    = config['d_model'],
    num_layers = config['num_layers'],
    num_heads  = config['num_heads'],
    d_ff       = config['d_ff'],
    dropout    = config['dropout'],
    max_length = config['max_length']
).to(DEVICE)

# Define the dataloader
train_loader = DataLoaderForLanguageModeling(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    sequence_length=80,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoaderForLanguageModeling(
    dataset=val_dataset,
    batch_size=config['batch_size'],
    sequence_length=80,
    shuffle=False,
    drop_last=True
)

# Define the criterion
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer =  torch.optim.Adam(model.parameters(recurse=True), lr=config['lr'])

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

# Define the scaler for mixed precision training
scaler = torch.amp.GradScaler(device=DEVICE)

# Print the model architecture and parameter summary
print(model)

# Optionally, if you want to summarize the model, make sure `torchsummaryX` is installed
summary = torchsummaryX.summary(model.to(DEVICE), x=next(iter(dl))[0].to(DEVICE))