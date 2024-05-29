from ast import mod
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, AutoImageProcessor


class WrappedHFModel(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, *args):
        return self.hf_model(*args).pooler_output


class WrappedHFPreprocessor:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def __call__(self, *args):
        return self.preprocessor(*args, return_tensors='pt')['pixel_values']


def load_clip_model(
        model_name: str,
        checkpoint_path: str | None = None,
        is_wrapped_checkpoint: bool = False,
        wrap_prefix: str | None = "model.",
    ):
    preprocessor = AutoImageProcessor.from_pretrained(model_name)
    preprocessor = WrappedHFPreprocessor(preprocessor)
    if checkpoint_path is None:
        model = CLIPVisionModel.from_pretrained(model_name)
        return WrappedHFModel(model), preprocessor

    cfg = CLIPVisionConfig.from_pretrained(model_name)
    model = CLIPVisionModel(cfg)

    chkpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in chkpt:
        state_dict = chkpt
    elif is_wrapped_checkpoint:
        state_dict = dict()
        for key, value in chkpt['state_dict'].items():
            if key.startswith(wrap_prefix):
                state_dict[key.replace(wrap_prefix, "")] = value
            elif key in model.state_dict():
                state_dict[key] = value
    else:
        state_dict = chkpt['state_dict']

    model.load_state_dict(state_dict)
    return WrappedHFModel(model), preprocessor
