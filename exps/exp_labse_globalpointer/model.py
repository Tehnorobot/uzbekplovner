"""LaBSE encoder and EfficientGlobalPointer span-classification head."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LABELS = ("ORG", "NAME", "GEO")


class EfficientGlobalPointer(nn.Module):
    """Score every valid start/end token pair for every entity label."""

    def __init__(self, hidden_size: int, num_labels: int, head_size: int = 64) -> None:
        super().__init__()
        if head_size < 2 or head_size % 2:
            raise ValueError("global_pointer.head_size must be a positive even integer")
        self.num_labels = num_labels
        self.head_size = head_size
        self.query_key = nn.Linear(hidden_size, head_size * 2)
        self.label_bias = nn.Linear(hidden_size, num_labels * 2)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inject relative token positions into query/key vectors with RoPE."""

        sequence_length = tensor.shape[1]
        frequencies = torch.arange(
            0,
            self.head_size,
            2,
            device=tensor.device,
            dtype=torch.float32,
        )
        frequencies = torch.pow(10000.0, -frequencies / self.head_size)
        positions = torch.arange(
            sequence_length,
            device=tensor.device,
            dtype=torch.float32,
        )
        sinusoid = torch.einsum("s,d->sd", positions, frequencies)
        sin = torch.repeat_interleave(sinusoid.sin(), 2, dim=-1)
        cos = torch.repeat_interleave(sinusoid.cos(), 2, dim=-1)
        sin = sin.to(dtype=tensor.dtype).unsqueeze(0)
        cos = cos.to(dtype=tensor.dtype).unsqueeze(0)
        rotated = torch.stack(
            (-tensor[..., 1::2], tensor[..., ::2]),
            dim=-1,
        ).reshape_as(tensor)
        return tensor * cos + rotated * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, labels, start, end]`` span logits."""

        query_key = self.query_key(hidden_states)
        query, key = query_key.chunk(2, dim=-1)
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        base_logits = torch.einsum("bmd,bnd->bmn", query, key)
        base_logits = base_logits / math.sqrt(self.head_size)

        bias = self.label_bias(hidden_states).transpose(1, 2) / 2
        start_bias = bias[:, 0::2, :, None]
        end_bias = bias[:, 1::2, None, :]
        logits = base_logits[:, None, :, :] + start_bias + end_bias

        pair_mask = span_mask[:, None, :, None] & span_mask[:, None, None, :]
        upper_triangle = torch.ones(
            logits.shape[-2:],
            dtype=torch.bool,
            device=logits.device,
        ).triu()
        pair_mask &= upper_triangle[None, None, :, :]
        return logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)


class LaBSEEfficientGlobalPointer(nn.Module):
    """Fine-tune a LaBSE encoder jointly with an EfficientGlobalPointer head."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        head_size: int,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        dropout_probability = (
            float(dropout)
            if dropout is not None
            else float(getattr(encoder.config, "hidden_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.pointer = EfficientGlobalPointer(hidden_size, num_labels, head_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch and calculate label-specific span logits."""

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        return self.pointer(self.dropout(encoder_output), span_mask)


def multilabel_categorical_crossentropy(
    targets: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Calculate stable multilabel categorical cross-entropy."""

    targets = targets.to(dtype=logits.dtype)
    logits = (1.0 - 2.0 * targets) * logits
    negative_logits = logits - targets * 1e12
    positive_logits = logits - (1.0 - targets) * 1e12
    zeros = torch.zeros_like(logits[..., :1])
    negative_logits = torch.cat((negative_logits, zeros), dim=-1)
    positive_logits = torch.cat((positive_logits, zeros), dim=-1)
    negative_loss = torch.logsumexp(negative_logits, dim=-1)
    positive_loss = torch.logsumexp(positive_logits, dim=-1)
    return (negative_loss + positive_loss).mean()


def global_pointer_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Flatten candidate spans and calculate the EfficientGlobalPointer loss."""

    batch_size, num_labels = logits.shape[:2]
    flat_labels = labels.reshape(batch_size, num_labels, -1)
    flat_logits = logits.float().reshape(batch_size, num_labels, -1)
    return multilabel_categorical_crossentropy(flat_labels, flat_logits)


def save_model_bundle(
    model: LaBSEEfficientGlobalPointer,
    tokenizer: Any,
    model_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Save encoder, pointer head, tokenizer and inference metadata locally."""

    model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(model_dir / "encoder")
    tokenizer.save_pretrained(model_dir / "tokenizer")
    torch.save(model.pointer.state_dict(), model_dir / "pointer_state.pt")
    (model_dir / "model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_bundle(
    model_dir: Path,
    device: torch.device,
) -> tuple[LaBSEEfficientGlobalPointer, Any, dict[str, Any]]:
    """Load a complete trained model from local artifacts without network access."""

    config_path = model_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model configuration: {config_path}")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    labels = tuple(metadata.get("labels", ()))
    if labels != LABELS:
        raise ValueError(f"unsupported saved labels: {labels!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer",
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with offset_mapping support is required")
    encoder = AutoModel.from_pretrained(
        model_dir / "encoder",
        local_files_only=True,
    )
    model = LaBSEEfficientGlobalPointer(
        encoder,
        num_labels=len(LABELS),
        head_size=int(metadata["head_size"]),
        dropout=float(metadata.get("dropout", 0.1)),
    )
    pointer_state = torch.load(
        model_dir / "pointer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.pointer.load_state_dict(pointer_state)
    model.to(device)
    model.eval()
    return model, tokenizer, metadata
