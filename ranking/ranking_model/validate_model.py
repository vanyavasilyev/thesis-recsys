import typing as tp
import pandas as pd
import tqdm

from .ranking_model_base import RankingModelBase, Reaction


def validate_ranking_model(
        model: RankingModelBase,
        reactions: pd.DataFrame,
        *,
        model_fit_kwargs=None,
        model_predict_kwargs=None,
        fit_every: int = 1,
        pretrain_share: float = 0.5,
        check_run_len: int | None = None):
    if model_fit_kwargs is None:
        model_fit_kwargs = {}
    if model_predict_kwargs is None:
        model_predict_kwargs = {}
    tt_split = int(len(reactions) * pretrain_share)
    train_reactions = reactions.iloc[0:tt_split]
    model.fit(
        [Reaction(*t) for t in train_reactions[["user_id", "image_id", "action_num", "created_at"]].itertuples(index=False)],
        model_fit_kwargs
    )
    gt = []
    pred = []

    i = 0
    cur_users = []
    cur_items = []
    cur_ts = []
    cur_values = []
    cur_reactions = []
    for row in tqdm.tqdm(
        list(
            reactions.iloc[tt_split:][["user_id", "image_id", "action_num", "created_at"]].itertuples(index=False))):
        if check_run_len is not None:
            if len(gt) > check_run_len:
                break 
        gt.append(row[2])
        cur_users.append(row[0])
        cur_items.append(row[1])
        cur_ts.append(row[3])
        cur_values.append(row[2])
        cur_reactions.append(Reaction(*row))
        i += 1
        if (i % fit_every == 0) or (i == len(reactions.iloc[tt_split:])):
            pred.extend(model.predict(cur_users, cur_items,
                                      cur_ts, cur_values,
                                      model_predict_kwargs))
            model.fit_partial(cur_reactions, model_fit_kwargs)
            cur_users = []
            cur_items = []
            cur_ts = []
            cur_values = []
            cur_reactions = []
    return gt, pred
