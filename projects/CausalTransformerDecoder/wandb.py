"""
Use Wandb for:
1. training model
2. saving checkpoints
3. hyperparameter tuning
4. monitor training performance
"""

USE_WANDB = False
RESUME_LOGGING = False

run_name = ""

if USE_WANDB:

    wandb.login(key="")

    if RESUME_LOGGING:
        run_id = ''
        run = wandb.init(
            settings=wandb.Settings(symlink=False),
            id     = run_id, ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
            project = "", ### Project should be created in your wandb account
        )
    else:
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "", ### Project should be created in your wandb account
            config  = config ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### log it in your wandb run with wandb.save()
        wandb.save('model_arch.txt')