import scipy.sparse
import typing as tp
import lightfm
import lightfm.data as ld
import lightfm.evaluation as lv

from .ranking_model_base import RankingModelBase, Reaction


class RankingLightFM(RankingModelBase):
    def __init__(self, users, looks, item_features_matrix, model_kwargs=None):
        super().__init__(users, looks, item_features_matrix, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        self.item_features_csr = scipy.sparse.csc_matrix(item_features_matrix)
        self.dataset = ld.Dataset()
        self.dataset.fit(users, looks)
        self.model = lightfm.LightFM(**model_kwargs)

    def fit(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        interactions, actions_matrix = self.dataset.build_interactions(
            [(r.user, r.item, r.value) for r in reactions])
        self.model = self.model.fit(actions_matrix, item_features=self.item_features_csr, **model_fit_kwargs)

    def fit_partial(self, reactions: tp.List[Reaction], model_fit_kwargs=None):
        if model_fit_kwargs is None:
            model_fit_kwargs = {}
        interactions, actions_matrix = self.dataset.build_interactions(
            [(r.user, r.item, r.value) for r in reactions])
        self.model = self.model.fit_partial(actions_matrix, item_features=self.item_features_csr, **model_fit_kwargs)

    def predict(self,
                users: str | tp.List[str],
                items: tp.List[str],
                timestamps: tp.List[tp.Any] | None = None,
                values:tp.List[tp.Any] | None = None,
                model_predict_kwargs=None,):
        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        if isinstance(users, str):
            users = [users]
        user_ids = [self.user2id[u] for u in users]
        look_ids = [self.look2id[l] for l in items]
        pred = self.model.predict(user_ids, look_ids, item_features=self.item_features_csr, **model_predict_kwargs)
        return pred
