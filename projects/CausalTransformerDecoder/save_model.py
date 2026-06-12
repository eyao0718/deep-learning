"""
Save model checkpoint here
"""

SAVE = False

# cannot upload saved model checkpoint since file > 25 MB
checkpoint_path = "./CausalLanguageModel_Checkpoint_X.pt"

class SaveModel():
    def save(self, model, optimizer, criterion, scheduler, scaler, max_epochs):
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "max_epochs": max_epochs,
                    "latest_epoch": epoch,
                    },
                    checkpoint_path)

if SAVE:
    checkpoint_save = SaveModel()
    checkpoint_save.save(model, optimizer, criterion, scheduler, scaler, config["num_epochs"])

