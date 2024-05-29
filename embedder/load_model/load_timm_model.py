import torch
import lightning as L
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def load_timm_model_from_checkpoint(
        model_name: str,
        checkpoint_path: str | None = None,
        num_classes: int = 0,
        is_wrapped_checkpoint: bool = False,
        wrap_prefix: str | None = "model.",
        load_head: bool = False,
        head_prefix: str | None = "head."):
    if checkpoint_path is None:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    chkpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in chkpt:
        state_dict = chkpt
    elif is_wrapped_checkpoint:
        state_dict = dict()
        for key, value in chkpt['state_dict'].items():
            if key.startswith(wrap_prefix + head_prefix):
                continue
            if key.startswith(wrap_prefix):
                state_dict[key.replace(wrap_prefix, "")] = value
            elif key in model.state_dict():
                state_dict[key] = value
    else:
        for key, value in chkpt['state_dict'].items():
            if key.startswith(head_prefix):
                continue
            state_dict[key] = value
    
    if not load_head:
        for key, value in model.state_dict().items():
            if key.startswith(head_prefix):
                state_dict[key] = value
    else:
        for key, value in chkpt['state_dict'].items():
            if key.startswith(head_prefix):
                state_dict[key] = value

    model.load_state_dict(state_dict)
    return model


def timm_transform(model):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform