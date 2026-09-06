"""XLM-RoBERTa encoder with a RoPE GlobalPointer span head."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LABELS = ("ORG", "NAME", "GEO")


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(tensor)


def _apply_rope(tensor: torch.Tensor) -> torch.Tensor:
    """Apply rotary positions to `[batch, sequence, labels, head]` tensors."""

    head_size = tensor.shape[-1]
    if head_size < 2 or head_size % 2:
        raise ValueError("global_pointer.head_size must be a positive even integer")
    positions = torch.arange(
        tensor.shape[1],
        device=tensor.device,
        dtype=torch.float32,
    )
    frequencies = torch.arange(0, head_size, 2, device=tensor.device, dtype=torch.float32)
    frequencies = torch.pow(10000.0, -frequencies / head_size)
    angles = torch.einsum("s,d->sd", positions, frequencies)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(tensor.dtype)[None, :, None]
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(tensor.dtype)[None, :, None]
    return tensor * cos + _rotate_half(tensor) * sin


class GlobalPointer(nn.Module):
    """Project contextual tokens into label-specific start/end span vectors."""

    def __init__(self, hidden_size: int, num_labels: int, head_size: int = 128) -> None:
        super().__init__()
        if head_size < 2 or head_size % 2:
            raise ValueError("global_pointer.head_size must be a positive even integer")
        self.num_labels = num_labels
        self.head_size = head_size
        self.projection = nn.Linear(hidden_size, num_labels * head_size * 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[batch, labels, start, end]` logits for valid spans."""

        batch_size, sequence_length, _ = hidden_states.shape
        projected = self.projection(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_labels,
            self.head_size * 2,
        )
        query, key = projected.chunk(2, dim=-1)
        query = _apply_rope(query)
        key = _apply_rope(key)
        logits = torch.einsum("bmhd,bnhd->bhmn", query, key)
        logits = logits / math.sqrt(self.head_size)

        pair_mask = span_mask[:, None, :, None] & span_mask[:, None, None, :]
        upper_triangle = torch.ones(
            logits.shape[-2:],
            dtype=torch.bool,
            device=logits.device,
        ).triu()
        pair_mask &= upper_triangle[None, None]
        return logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)


class XLMRobertaGlobalPointer(nn.Module):
    """Fine-tune XLM-RoBERTa jointly with the GlobalPointer head."""

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
        self.pointer = GlobalPointer(hidden_size, num_labels, head_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch and calculate span logits."""

        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.pointer(self.dropout(hidden_states), span_mask)


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
    negative_loss = torch.logsumexp(torch.cat((negative_logits, zeros), dim=-1), dim=-1)
    positive_loss = torch.logsumexp(torch.cat((positive_logits, zeros), dim=-1), dim=-1)
    return (negative_loss + positive_loss).mean()


def global_pointer_loss(targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Flatten candidate spans and calculate GlobalPointer loss."""

    batch_size, num_labels = logits.shape[:2]
    flat_targets = targets.reshape(batch_size, num_labels, -1)
    flat_logits = logits.float().reshape(batch_size, num_labels, -1)
    return multilabel_categorical_crossentropy(flat_targets, flat_logits)


def save_model_bundle(
    model: XLMRobertaGlobalPointer,
    tokenizer: Any,
    model_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Save encoder, GlobalPointer head, tokenizer and inference metadata."""

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
) -> tuple[XLMRobertaGlobalPointer, Any, dict[str, Any]]:
    """Load a complete model bundle without accessing external services."""

    config_path = model_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model configuration: {config_path}")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    labels = tuple(metadata.get("labels", ()))
    if labels != LABELS:
        raise ValueError(f"unsupported saved labels: {labels!r}")
    if metadata.get("architecture") != "XLMRobertaGlobalPointer":
        raise ValueError(f"unsupported saved architecture: {metadata.get('architecture')!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer",
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with offset_mapping support is required")
    encoder = AutoModel.from_pretrained(model_dir / "encoder", local_files_only=True)
    model = XLMRobertaGlobalPointer(
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
