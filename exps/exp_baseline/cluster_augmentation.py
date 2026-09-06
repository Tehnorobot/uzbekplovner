"""Cluster-based, label-preserving augmentation for the NER baseline."""

import hashlib
import random
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as functional

from .common import ENTITY_LABELS, JsonObject


def _entity_surfaces(records: list[JsonObject]) -> dict[str, list[str]]:
    """Collect unique entity surfaces separately for every target label."""

    surfaces: dict[str, set[str]] = {label: set() for label in ENTITY_LABELS}
    for record in records:
        text = record["text"]
        for entity in record["entities"]:
            surface = text[entity["start"] : entity["end"]]
            if surface.strip():
                surfaces[entity["label"]].add(surface)
    return {label: sorted(values) for label, values in surfaces.items()}


def _surface_vectors(
    surfaces: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    *,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Represent each surface by the mean of its input-token embeddings."""

    if not surfaces:
        return {}
    embedding = model.get_input_embeddings()
    weight = embedding.weight.detach()
    vectors: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(surfaces), batch_size):
            batch_surfaces = surfaces[start : start + batch_size]
            encoded = tokenizer(
                batch_surfaces,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=64,
                return_attention_mask=True,
            )
            input_ids = torch.as_tensor(encoded["input_ids"], device=weight.device)
            attention_mask = torch.as_tensor(
                encoded["attention_mask"], device=weight.device, dtype=weight.dtype
            )
            token_vectors = weight[input_ids].float()
            pooled = (token_vectors * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            vectors.extend(functional.normalize(pooled, dim=1).cpu())
    return dict(zip(surfaces, vectors, strict=True))


def _kmeans(vectors: torch.Tensor, cluster_count: int, seed: int) -> list[int]:
    """Cluster normalized vectors with deterministic cosine K-Means."""

    if len(vectors) == 0:
        return []
    cluster_count = min(cluster_count, len(vectors))
    if cluster_count == 1:
        return [0] * len(vectors)

    generator = torch.Generator()
    generator.manual_seed(seed)
    initial_indices = torch.randperm(len(vectors), generator=generator)[:cluster_count]
    centroids = vectors[initial_indices].clone()
    for _ in range(25):
        assignments = (vectors @ centroids.T).argmax(dim=1)
        updated = []
        for cluster_index in range(cluster_count):
            members = vectors[assignments == cluster_index]
            centroid = members.mean(dim=0) if len(members) else centroids[cluster_index]
            updated.append(functional.normalize(centroid.unsqueeze(0), dim=1).squeeze(0))
        new_centroids = torch.stack(updated)
        if torch.equal(assignments, (vectors @ new_centroids.T).argmax(dim=1)):
            centroids = new_centroids
            break
        centroids = new_centroids
    return (vectors @ centroids.T).argmax(dim=1).tolist()


def _cluster_surfaces(
    surfaces_by_label: dict[str, list[str]],
    vectors: dict[str, torch.Tensor],
    *,
    num_clusters: int,
    seed: int,
) -> dict[tuple[str, int], list[str]]:
    """Build label-specific clusters and return their candidate dictionaries."""

    clusters: dict[tuple[str, int], list[str]] = defaultdict(list)
    for label_index, label in enumerate(ENTITY_LABELS):
        surfaces = surfaces_by_label[label]
        if not surfaces:
            continue
        matrix = torch.stack([vectors[surface] for surface in surfaces])
        # Keep more than one candidate in a cluster whenever the dictionary allows it.
        cluster_count = min(num_clusters, max(1, len(surfaces) // 2))
        assignments = _kmeans(matrix, cluster_count, seed + label_index)
        for surface, cluster_index in zip(surfaces, assignments, strict=True):
            clusters[(label, cluster_index)].append(surface)
    return dict(clusters)


def _replacement(
    source: str,
    label: str,
    vectors: dict[str, torch.Tensor],
    clusters: dict[tuple[str, int], list[str]],
    assignments: dict[tuple[str, str], int],
    *,
    max_candidates: int,
    randomizer: random.Random,
) -> str | None:
    """Select a similar same-label surface from the source surface's cluster."""

    cluster_index = assignments.get((label, source))
    if cluster_index is None:
        return None
    candidates = [
        candidate for candidate in clusters.get((label, cluster_index), []) if candidate != source
    ]
    if not candidates:
        return None
    source_vector = vectors[source]
    candidates.sort(
        key=lambda candidate: float(source_vector @ vectors[candidate]),
        reverse=True,
    )
    return randomizer.choice(candidates[:max_candidates])


def _make_augmented_record(
    record: JsonObject,
    replacements: list[tuple[JsonObject, str]],
    variant_index: int,
) -> JsonObject:
    """Replace selected surfaces while preserving and realigning every entity."""

    text = record["text"]
    parts: list[str] = []
    entities: list[JsonObject] = []
    cursor = 0
    new_cursor = 0
    replacements_by_entity = {
        (entity["label"], entity["start"], entity["end"]): replacement
        for entity, replacement in replacements
    }
    for entity in sorted(
        record["entities"],
        key=lambda item: (item["start"], item["end"], item["label"]),
    ):
        start, end = entity["start"], entity["end"]
        parts.append(text[cursor:start])
        new_cursor += len(text[cursor:start])
        entity_start = new_cursor
        surface = replacements_by_entity.get(
            (entity["label"], start, end),
            text[start:end],
        )
        parts.append(surface)
        new_cursor += len(surface)
        entities.append({"label": entity["label"], "start": entity_start, "end": new_cursor})
        cursor = end
    parts.append(text[cursor:])
    augmented_text = "".join(parts)
    digest = hashlib.sha256(
        f"{record['hash']}\0cluster\0{variant_index}\0{augmented_text}".encode()
    ).hexdigest()
    return {
        "hash": f"cluster-{digest}",
        "text": augmented_text,
        "entities": entities,
        "augmentation": {
            "method": "cluster_based",
            "source_hash": record["hash"],
            "variant": variant_index,
        },
    }


def apply_cluster_augmentation(
    records: list[JsonObject],
    tokenizer: Any,
    model: torch.nn.Module,
    *,
    enabled: bool,
    probability: float,
    num_clusters: int,
    max_candidates: int,
    seed: int,
) -> tuple[list[JsonObject], dict[str, Any]]:
    """Append one cluster-based variant for selected training records."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation probability must be in [0, 1]")
    if num_clusters < 1:
        raise ValueError("augmentation num_clusters must be positive")
    if max_candidates < 1:
        raise ValueError("augmentation max_candidates must be positive")

    stats: dict[str, Any] = {
        "enabled": enabled,
        "probability": probability,
        "num_clusters": num_clusters,
        "max_candidates": max_candidates,
        "seed": seed,
        "input_records": len(records),
        "added_records": 0,
        "replaced_entities": 0,
    }
    if not enabled or probability == 0.0:
        stats["reason"] = "disabled" if not enabled else "zero_probability"
        return records, stats

    surfaces_by_label = _entity_surfaces(records)
    all_surfaces = [surface for values in surfaces_by_label.values() for surface in values]
    vectors = _surface_vectors(all_surfaces, tokenizer, model)
    clusters = _cluster_surfaces(
        surfaces_by_label,
        vectors,
        num_clusters=num_clusters,
        seed=seed,
    )
    assignments = {
        (label, surface): cluster_index
        for (label, cluster_index), surfaces in clusters.items()
        for surface in surfaces
    }
    randomizer = random.Random(seed)
    augmented = list(records)
    variant_index = 0
    for record in records:
        if randomizer.random() >= probability:
            continue
        replacements: list[tuple[JsonObject, str]] = []
        for entity in record["entities"]:
            source = record["text"][entity["start"] : entity["end"]]
            replacement = _replacement(
                source,
                entity["label"],
                vectors,
                clusters,
                assignments,
                max_candidates=max_candidates,
                randomizer=randomizer,
            )
            if replacement is not None:
                replacements.append((entity, replacement))
        if not replacements:
            continue
        variant_index += 1
        augmented.append(_make_augmented_record(record, replacements, variant_index))
        stats["added_records"] += 1
        stats["replaced_entities"] += len(replacements)

    stats["output_records"] = len(augmented)
    stats["cluster_count"] = len(clusters)
    return augmented, stats
