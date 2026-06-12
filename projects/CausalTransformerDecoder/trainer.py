"""
Trainer Class
"""

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, max_epochs=1):
        """
            Use this class to train your model
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler

        self.train_losses = []
        self.val_losses = []
        self.prediction_probs = []
        self.prediction_probs_test = []
        self.generated_texts_test = []
        self.generated_texts_test_beam = []
        self.generated_texts_test_beam_random = []
        self.generated_texts_validation = []

        self.log_likelihood_beam = []
        self.log_likelihood_beam_random = []

        self.train_epochs = 0
        self.val_epochs = 0
        self.max_epochs = max_epochs

    def calculate_loss(self, input, target):
        """
        Crossentropy calculates the loss between a label and its probability distribution
        
        Args:
            input: probability distributions (B, T, Vocab_size)
            target: class indices (B, T)

        Returns:
            cross entropy loss (torch.tensor)
        """
        C = input.shape[-1]
        loss = self.criterion(input.view(-1, C), target.view(-1))
        return loss

    def train(self):
        """
        Training for a single epoch
        
        Returns:
            performance: (epoch_loss, lr, attn_weights)
        """
        # set to training mode
        self.model.train()
        self.model.to(DEVICE)
        loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        attn_weights = None
        for batch_num, (inputs, targets) in enumerate(tqdm(self.train_loader)):

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):

                # forward pass
                outputs, attn_weights = self.model.forward(inputs)

                # calculate loss
                loss = self.calculate_loss(outputs, targets)
            
            # zeroes the gradients to prevent accumulation after each batch
            self.optimizer.zero_grad()

            """Use Mixed Precision Training"""
            # multiplies the loss by a large scaling factor, then backpropagate.
            # gradients are computed from the scaled loss, then corrected later.
            # prevents gradient underflow using FP16
            self.scaler.scale(loss).backward()
            # loss.backward()
            # 1. scaler unscales the gradients
            # 2. if NOT inf/nan gradient, calls optimizer.step(). else skips update step for gradient
            # learnable parameters get updated because optimizer.step() updates them after each batch
            self.scaler.step(self.optimizer)
            # self.optimizer.step()

            # After each iteration, the scaler adjusts its scaling factor dynamically.
            # 1. If training stable, it increases the scale factor.
            # 2. If overflow detected, it decreases the scale factor.
            self.scaler.update()
    
            # Update epoch loss
            epoch_loss += loss.item()
            num_batches += 1
        
        # average epoch loss over total number of batches
        # epoch_loss = average batch loss for this epoch
        epoch_loss /= num_batches
        self.train_epochs += 1
        # the learning rate is going to change because scheduler.step() changes it after each epoch
        print('[TRAIN] \tEpoch [%d/%d] \tLoss: %.4f \tLr: %.6f'
              % (self.train_epochs + curr_ep, self.max_epochs, epoch_loss, self.optimizer.param_groups[0]['lr']))
        
        self.train_losses.append(epoch_loss)

        return (epoch_loss, self.optimizer.param_groups[0]['lr'], attn_weights)

    def validate(self):
        # set to eval mode
        self.model.eval()
        self.model.to(DEVICE)
        loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        # No gradients needed for validation
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(tqdm(self.val_loader)):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs, attn_weights = self.model(inputs)
                loss = self.calculate_loss(outputs, targets)
                epoch_loss += loss.item()
                num_batches += 1
        epoch_loss /= num_batches
        self.val_epochs += 1
        print('[VAL] \tEpoch [%d/%d] \tLoss: %.4f \tLr: %.6f'
              % (self.val_epochs + curr_ep, self.max_epochs, epoch_loss, self.optimizer.param_groups[0]['lr']))
        self.val_losses.append(epoch_loss)
        return epoch_loss



# trainer class object takes in everything
trainer = Trainer(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    optimizer = optimizer,
    criterion = criterion,
    scheduler = scheduler,
    scaler = scaler,
    max_epochs = config["num_epochs"], # TODO: set the number of epochs
)