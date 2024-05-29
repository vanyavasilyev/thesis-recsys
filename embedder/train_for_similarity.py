import pandas as pd
import json
import torch
import lightning as L
from torch.utils.data import DataLoader

from utils import LitSimilarityModel, TripletDataset
from load_model import load_model

torch.set_float32_matmul_precision("medium")

with open("configs/similarity.json", "r") as f:
    config = json.load(f)

model, transform = load_model(**config["model_args"])

triplets_df = pd.read_csv(config["data_location"]["triplets_file"])
splits = ["train", "val"]
keys = ["anchor", "positive", "negative"]
datasets = {
    split: TripletDataset(
        config["data_location"]["data_dir"],
        triplets_df[triplets_df['split'] == split][keys],
        transform,
        full_paths=config["data_location"]["triplets_full_paths"])
    for split in splits
}
dataloaders = {
    split: DataLoader(
        datasets[split],
        shuffle=(split=="train"),
        **config["dataloader_args"])
    for split in splits
}


first_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=config["data_location"]["checkpoint_dir_name"],
    filename="worst", save_top_k=1, monitor="val_loss", mode="max")
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=config["data_location"]["checkpoint_dir_name"],
    save_top_k=10, monitor="val_loss")
checkpoints = [checkpoint_callback]
if config["misc"]["save_first"]:
    checkpoints.append(first_checkpoint_callback)

trainer = L.Trainer(callbacks=checkpoints, **config["trainer_args"])
trainer.fit(LitSimilarityModel(model, **config["lightning_model_args"]),
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['val'],
            )
