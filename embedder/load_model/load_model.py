from .load_timm_model import load_timm_model_from_checkpoint, timm_transform
from .load_hf_model import load_clip_model


def load_model(
        model_name: str,
        checkpoint_path: str | None = None,
        num_classes: int = 0,
        is_wrapped_checkpoint: bool = False,
        wrap_prefix: str | None = "model.",
        load_head: bool = False,
        head_prefix: str | None = "head.",
        model_type: str = "timm_model"):
    if model_type == "timm_model":
        model = load_timm_model_from_checkpoint(
            model_name, checkpoint_path, num_classes, is_wrapped_checkpoint,
            wrap_prefix, load_head, head_prefix
        )
        transform = timm_transform(model)
        return model, transform
    if model_type == 'clip':
        return load_clip_model(
            model_name,
            checkpoint_path,
            is_wrapped_checkpoint,
            wrap_prefix,
        )
