import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE
from config import *

# Load training dataset
with open(TRAIN_DATASET, "rb") as f:
    training = pickle.load(f)

train_loader = DataLoader(training, batch_size=8, shuffle=True, num_workers=0)

# Trainer
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="cpu",         # do NOT use GPU on your laptop
    enable_checkpointing=False
)

# Build TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    loss=SMAPE()
)

# NEW Lightning 2.x syntax:
trainer.fit(
    model=tft,
    train_dataloaders=train_loader
)

# Save model
tft.save(TFT_MODEL_PATH)
print("Model saved to:", TFT_MODEL_PATH)
