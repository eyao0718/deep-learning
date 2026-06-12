"""
Experiments:

Run the experiments loop.
Each epoch wont take more than 2-3min.
If its taking more time,
  * You might be overlapping batches
      Eg. Input: "I had biryani for lunch today" and sequence length = 3,
          --> "I had biryani", "for lunch today" are ideal examples for inputs
          --> "I had biryani", "had biryani for", "biryani for lunch", ... is just redundant info :')
  * Your length calculation in the dataloader might be wrong
"""

# wandb.watch(model, log="all")

torch.mps.empty_cache()
gc.collect()

for epoch in range(curr_ep, config["num_epochs"]):
    train_loss, curr_lr,  attn_weights = trainer.train()
    test_loss = trainer.validate()
    # wandb.log({"train_loss":train_loss,
    #            "test_loss": test_loss,
    #            "learning_rate": curr_lr
    #           })
    scheduler.step()

# Finish your wandb run
# run.finish()

torch.mps.empty_cache()
gc.collect()