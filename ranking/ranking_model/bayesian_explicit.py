import typing as tp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .ranking_model_base import RankingModelBase, Reaction


class ReactionsDataset(Dataset):
    def __init__(self, reactions: tp.List[Reaction], user2id: tp.Dict[str, int], look2id: tp.Dict[str, int]):
        super().__init__()
        self.reactions = reactions
        self.user2id = user2id
        self.look2id = look2id

    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, index):
        r = self.reactions[index]
        return self.user2id[r.user], self.look2id[r.item], r.value


class ExplicitBayesian(nn.Module):
    def __init__(self, num_users, num_looks, item_features):
        super().__init__()
        self.item_embeddings = nn.Embedding.from_pretrained(torch.tensor(item_features))
        self.user_embeddings = nn.Embedding(num_users, item_features.shape[1])

    def forward(self, users, items):
        with torch.no_grad():
            ie = self.item_embeddings(items)
        ue = self.user_embeddings(users)
        logits = torch.sum(ue * ie, dim=-1)
        return logits


class RankingBayesian(RankingModelBase):
    def __init__(self, users, looks, item_features_matrix, model_kwargs=None):
        super().__init__(users, looks, item_features_matrix, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        self.device = torch.device("cpu")
        if "device" in model_kwargs:
            self.device = torch.device(model_kwargs["device"])
        self.model = ExplicitBayesian(len(users), len(looks), item_features_matrix).to(self.device)
        self.reactions = []

    @staticmethod
    def _get_fit_args(model_fit_kwargs):
        res = {
            "batch_size": 32,
            "max_epochs": 10,
        }
        res.update(model_fit_kwargs)
        return res

    def _fit(self, model_fit_kwargs, new_users=None):
        fit_kwargs_full = self._get_fit_args(model_fit_kwargs)
        if new_users is None:
            training_reactions = self.reactions
        else:
            training_reactions = []
            for r in self.reactions:
                if r.user in new_users:
                    training_reactions.append(r)
        dataset = ReactionsDataset(training_reactions, self.user2id, self.look2id)
        dataloader = DataLoader(dataset, fit_kwargs_full["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_history = []
        for i in range(fit_kwargs_full["max_epochs"]):
            epoch_loss = []
            for batch in dataloader:
                optimizer.zero_grad()
                users, items, values = batch
                values = ((values + 1) / 2).to(torch.long)
                logits = self.model(users.to(self.device), items.to(self.device)).cpu()
                logits = torch.stack((torch.zeros_like(logits), logits), dim=1)
                loss = F.cross_entropy(logits, values)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            loss_history.append(np.mean(epoch_loss))
            if np.argmin(loss_history) < len(loss_history) - 4:
                break
        # print(loss_history)


    def fit(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        self.reactions = reactions
        self._fit(model_fit_kwargs)

    def fit_partial(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        new_users = set()
        for r in reactions:
            new_users.add(r.user)
        self.reactions.extend(reactions)
        self._fit(model_fit_kwargs, new_users)

    def predict(self,
                users: str | tp.List[str],
                items: tp.List[str],
                timestamps: tp.List[tp.Any] | None = None,
                values:tp.List[tp.Any] | None = None,
                model_predict_kwargs=None,):
        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        if isinstance(users, str):
            user_id = [self.user2id[users] for l in items]
        else:
            user_id = [self.user2id[u] for u in users]
        look_ids = [self.look2id[l] for l in items]
        with torch.no_grad():
            logits = self.model(torch.tensor(user_id).to(self.device), 
                                torch.tensor(look_ids).to(self.device)).cpu()
            return F.sigmoid(logits).numpy()
        
