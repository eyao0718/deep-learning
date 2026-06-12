"""
Load model checkpoint here
"""

LOAD = False

# cannot upload saved model checkpoint since file > 25 MB
checkpoint_path = "./CausalLanguageModel_Checkpoint_X.pt"

curr_ep = 35 # TODO: Set epoch you left off

class LoadModel():

    def load(self, path):

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        curr_epoch = checkpoint["latest_epoch"]
        max_epoch = checkpoint["max_epochs"]


if LOAD:
    checkpoint_load = LoadModel()
    checkpoint_load.load(checkpoint_path)