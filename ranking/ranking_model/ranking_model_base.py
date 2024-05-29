import typing as tp

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Reaction:
    user: str
    item: str
    value: float
    timestamp: tp.Any


class RankingModelBase(ABC):
    @abstractmethod
    def __init__(self, users, looks, item_features_matrix, model_kwargs=None):
        self.users = users
        self.user2id = {users[i]: i for i in range(len(users))}
        self.looks = looks
        self.look2id = {looks[i]: i for i in range(len(looks))}
        self.item_features_matrix = item_features_matrix

    @abstractmethod
    def fit(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        pass

    @abstractmethod
    def fit_partial(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        pass

    @abstractmethod
    def predict(self,
                users: str | tp.List[str],
                items: tp.List[str],
                timestamps: tp.List[tp.Any] | None = None,
                values:tp.List[tp.Any] | None = None,
                model_predict_kwargs=None,):
        pass
