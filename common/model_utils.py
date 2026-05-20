from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor



def load_processor(cfg: dict[str, Any]) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        cfg["model_name"],
        language=cfg["language"],
        task=cfg["task"],
    )


def load_processor_from_dir(model_dir: str | Path, cfg: dict[str, Any]) -> WhisperProcessor:
    model_dir = Path(model_dir)
    if model_dir.exists():
        return WhisperProcessor.from_pretrained(
            str(model_dir),
            language=cfg["language"],
            task=cfg["task"],
        )
    return load_processor(cfg)



def load_model(cfg: dict[str, Any]) -> WhisperForConditionalGeneration:
    quant = cfg.get("model_weight_quantization")
    use_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {}

    if quant == "8bit" and use_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_name"], **kwargs)

    if quant != "8bit":
        model = model.to(torch.float32)

    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    return model


def load_trained_model(model_dir: str | Path, cfg: dict[str, Any]):
    model_dir = Path(model_dir)
    adapter_config = model_dir / "adapter_config.json"

    if adapter_config.exists():
        from peft import PeftModel

        base_cfg = dict(cfg)
        base_cfg["model_weight_quantization"] = None
        base_model = load_model(base_cfg)
        model = PeftModel.from_pretrained(base_model, str(model_dir))
    else:
        dtype = torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")

    generation_owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    generation_owner.generation_config.language = cfg["language"]
    generation_owner.generation_config.task = cfg["task"]
    model.eval()
    return model



def apply_full_ft(model, cfg: dict[str, Any]):
    model = model.float()
    for param in model.parameters():
        param.requires_grad = True
    return model



def apply_selective_ft(model, cfg: dict[str, Any]):
    model = model.float()

    freeze_enc = int(cfg.get("freeze_encoder_bottom_n_layers", 0))
    freeze_dec = int(cfg.get("freeze_decoder_bottom_n_layers", 0))

    for param in model.parameters():
        param.requires_grad = True

    encoder_layers = model.model.encoder.layers
    decoder_layers = model.model.decoder.layers

    for layer in encoder_layers[:freeze_enc]:
        for param in layer.parameters():
            param.requires_grad = False

    for layer in decoder_layers[:freeze_dec]:
        for param in layer.parameters():
            param.requires_grad = False

    return model



def apply_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.get("model_weight_quantization") == "8bit" and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_cfg)
    return model



def configure_model_for_strategy(model, cfg: dict[str, Any]):
    strategy = cfg["strategy_name"]
    if strategy == "full_ft":
        return apply_full_ft(model, cfg)
    if strategy == "selective_ft":
        return apply_selective_ft(model, cfg)
    if strategy == "lora_ft":
        return apply_lora(model, cfg)
    raise ValueError(f"Nežinoma strategija: {strategy}")



def count_trainable_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}
