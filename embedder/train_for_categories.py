import pandas as pd
import numpy as np
import torch
import json
import lightning as L
from torch.utils.data import DataLoader

from utils import LitCategoryModel, LitCategoryModelWithHead, CategoryDataset
from load_model import load_model

torch.set_float32_matmul_precision("medium")
L.seed_everything(2929)
with open("configs/category.json", "r") as f:
    config = json.load(f)

with open(config["data_location"]["base_categories_file"], "r") as f:
    base_categories = json.load(f)
with open(config["data_location"]["categories_mapping_file"], "r") as f:
    categories_mapping = json.load(f)

# config["model_args"]["num_classes"] = len(base_categories)
config["lightning_model_args"]["loss_kwargs"]["num_classes"] = len(base_categories)

model, transform = load_model(**config["model_args"])

categories_df = pd.read_csv(config["data_location"]["crop_categories_file"])
splits = ["train", "val"]
datasets = {
    split: CategoryDataset(
        categories_df[categories_df['split'] == split],
        base_categories,
        categories_mapping,
        transform)
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
    filename="worst", save_top_k=1, monitor="val_loss", mode="max"
    )
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=config["data_location"]["checkpoint_dir_name"],
    save_top_k=2, monitor="val_loss"
    )
checkpoints = [checkpoint_callback]
if config["misc"]["save_first"]:
    checkpoints.append(first_checkpoint_callback)

trainer = L.Trainer(callbacks=checkpoints, **config["trainer_args"])
trainer.fit(LitCategoryModelWithHead(model, **config["lightning_model_args"]),
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['val'],
            )
