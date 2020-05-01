import nsp

"""
This is sample showing how to train model
"""
# Load train manager
trainer = nsp.manager.TrainManager("data/train_config.yml", "cpu")

# Start training
trainer.train()
