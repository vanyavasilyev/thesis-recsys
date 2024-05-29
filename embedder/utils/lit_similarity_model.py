import torch
import torch.nn.functional as F
import lightning as L


class LitSimilarityModel(L.LightningModule):
    def __init__(self, model, lr=1e-4, margin=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.margin = margin

    def _get_loss(self, batch):
        a, p, n = batch
        a_embeds = self.model(a.squeeze().contiguous())
        p_embeds = self.model(p.squeeze().contiguous())
        n_embeds = self.model(n.squeeze().contiguous())
        loss = F.triplet_margin_with_distance_loss(a_embeds, p_embeds, n_embeds,
                                                   distance_function=lambda t1, t2: -F.cosine_similarity(t1, t2, dim=-1),
                                                   margin=self.margin)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss  

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}