import typing as tp
import numpy as np
import pandas as pd
import math
import copy
from requests import session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from traitlets import Long

from .ranking_model_base import RankingModelBase, Reaction


class SessionsDataset(Dataset):
    def __init__(self, sessions, values):
        super().__init__()
        self.sessions = sessions
        self.values = values

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, index):
        return np.stack(self.sessions[index]), self.values[index]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe
        return self.dropout(x)


class SequentialRNN(nn.Module):
    def __init__(self, rnn_kwargs):
        super().__init__()
        self.rnn = nn.LSTM(**rnn_kwargs)
        self.layer_size = rnn_kwargs["hidden_size"]
        if rnn_kwargs["bidirectional"]:
            self.layer_size *= 2
        self.head = nn.Linear(self.layer_size, 1)

    def forward(self, sessions):
        rnn_out, _ = self.rnn(sessions)
        logits = self.head(rnn_out[:,0,:])
        return logits
        

class SequentialTransformer(nn.Module):
    def __init__(self,
                 inp_dim,
                 d_model,
                 seq_len,
                 nhead,
                 nlayers):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.to_model_dim = nn.Linear(inp_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, sessions):
        x = self.to_model_dim(sessions)
        x = self.pos(x)
        x = self.transformer_encoder(x).sum(1)
        logits = self.head(x)
        return logits



class RankingSequential(RankingModelBase):
    def __init__(self, users, looks, item_features_matrix, model_kwargs=None):
        super().__init__(users, looks, item_features_matrix, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        init_args = self._get_init_args(model_kwargs)
        self.device = torch.device(init_args["device"])
        self.retrain_every = init_args["retrain_every"]
        self.session_len = init_args["session_len"]

        self.item_features = np.concatenate((item_features_matrix, np.zeros((1, item_features_matrix.shape[1]))))
        self.last_reactions = defaultdict(list)
        self.sessions = []
        self.values = []
        self.partial_fits = 0
        for key in init_args:
            if key in model_kwargs:
                del model_kwargs[key]
        if init_args["model_type"] == "rnn":
            model_kwargs["input_size"] = self.item_features.shape[1] + 1
            rnn_kwargs = self._get_rnn_args(model_kwargs)
            self.model = SequentialRNN(rnn_kwargs).to(self.device)
        if init_args["model_type"] == "tr":
            inp_dim = self.item_features.shape[1] + 1
            seq_len = self.session_len + 1
            tr_kwargs = self._get_tr_args(model_kwargs)
            self.model = SequentialTransformer(
                inp_dim=inp_dim,
                d_model=tr_kwargs["d_model"],
                seq_len=seq_len,
                nhead=tr_kwargs["nhead"],
                nlayers=tr_kwargs["nlayers"],
            ).to(self.device)

    @staticmethod
    def _get_init_args(model_kwargs):
        res = {
            "device": "cpu",
            "session_len": 10,
            "retrain_every": 10,
            "model_type": "rnn",
        }
        res.update(model_kwargs)
        return res

    @staticmethod
    def _get_rnn_args(model_kwargs):
        res = {
            "hidden_size": 128,
            "num_layers": 4,
            "batch_first": True,
            "bidirectional": True,
        }
        res.update(model_kwargs)
        return res

    @staticmethod
    def _get_tr_args(model_kwargs):
        res = {
            "d_model": 128,
            "nhead": 4,
            "nlayers": 4,
        }
        res.update(model_kwargs)
        return res

    @staticmethod
    def _get_fit_args(model_fit_kwargs):
        res = {
            "batch_size": 128,
            "max_epochs": 10,
        }
        res.update(model_fit_kwargs)
        return res
    
    @staticmethod
    def _get_predict_args(model_predict_kwargs):
        res = {
            "batch_size": 128,
        }
        res.update(model_predict_kwargs)
        return res

    def _fit(self, sessions, values, model_fit_kwargs):
        self.model.train()
        fit_kwargs_full = self._get_fit_args(model_fit_kwargs)
        dataset = SessionsDataset(sessions, values)
        dataloader = DataLoader(dataset, fit_kwargs_full["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_history = []
        for i in range(fit_kwargs_full["max_epochs"]):
            epoch_loss = []
            for batch in dataloader:
                optimizer.zero_grad()
                batch_sessions, batch_values = batch
                batch_values = ((batch_values + 1) / 2).to(torch.long)
                logits = self.model(batch_sessions.to(torch.float32).to(self.device)).cpu()
                logits = torch.concat((torch.zeros_like(logits), logits), dim=1)
                loss = F.cross_entropy(logits, batch_values)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            loss_history.append(np.mean(epoch_loss))
            if np.argmin(loss_history) < len(loss_history) - 2:
                break

    def _vector_from_reaction(self, reaction: Reaction, value=None):
        if value is None:
            value = reaction.value
        vector = self.item_features[self.look2id[reaction.item],:]
        vector = np.concatenate([vector, np.array([value])])
        return vector

    def _session_from_old_reactions(self, old_reations: tp.List[Reaction], new_reaction: Reaction):
        vectors = []
        vectors.append(self._vector_from_reaction(new_reaction, 0.0))
        for i in range(min(len(old_reations), self.session_len)):
            vectors.append(self._vector_from_reaction(old_reations[-i-1]))
        while len(vectors) < self.session_len + 1:
            vectors.append(
                np.concatenate((self.item_features[len(self.looks),:], np.array([0.0]))))
        return vectors

    def _new_sessions(self, reactions: tp.List[Reaction], add: bool=True):
        new_sessions = []
        new_values = []
        for r in reactions:
            new_values.append(r.value)
            old_reactions = self.last_reactions[r.user]
            if len(old_reactions) > 0:
                last_reaction = old_reactions[-1]
                if pd.Timestamp(r.timestamp) - pd.Timestamp(last_reaction.timestamp) > pd.Timedelta(30, "min"):
                    self.last_reactions[r.user] = []
                    old_reactions = []
            session = self._session_from_old_reactions(old_reactions, r)
            new_sessions.append(session)
            self.last_reactions[r.user].append(r)
        if add:
            self.sessions.extend(new_sessions)
            self.values.extend(new_values)
        return new_sessions, new_values

    def fit(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        self.sessions = []
        self.values = []
        self._new_sessions(reactions)
        self._fit(self.sessions, self.values, model_fit_kwargs)

    def fit_partial(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        sessions, values = self._new_sessions(reactions)
        if self.partial_fits % self.retrain_every:
            self._fit(sessions, values, model_fit_kwargs)
        else:
            self._fit(self.sessions, self.values, model_fit_kwargs)
        self.partial_fits += 1

    def predict(self,
                users: str | tp.List[str],
                items: tp.List[str],
                timestamps: tp.List[tp.Any] | None = None,
                values:tp.List[tp.Any] | None = None,
                model_predict_kwargs=None,):
        self.model.eval()
        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        predict_kwargs_full = self._get_predict_args(model_predict_kwargs)
        if isinstance(users, str):
            user_id = [self.user2id[users] for l in items]
        else:
            user_id = [self.user2id[u] for u in users]
        look_ids = [self.look2id[l] for l in items]
        last_reactions_old = copy.deepcopy(self.last_reactions)
        reactions = []
        for i in range(len(users)):
            reactions.append(Reaction(
                users[i],
                items[i],
                values[i],
                timestamps[i]
            ))
        new_sessions, new_values = self._new_sessions(reactions, False)
        dataset = SessionsDataset(new_sessions, new_values)
        dataloader = DataLoader(dataset, predict_kwargs_full["batch_size"], shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                sessions, _ = batch
                preds.extend(F.sigmoid(self.model(sessions.to(torch.float32).to(self.device))).cpu().tolist())
        self.last_reactions = last_reactions_old
        return preds    
