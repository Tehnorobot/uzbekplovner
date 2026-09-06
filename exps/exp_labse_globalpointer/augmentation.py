"""Optional label-preserving cluster augmentation for entity surfaces."""

import hashlib
import random
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as functional

from .model import LABELS

JsonObject = dict[str, Any]


def _entity_surfaces(records: list[JsonObject]) -> dict[str, list[str]]:
    """Collect unique entity surfaces independently for every label."""

    surfaces: dict[str, set[str]] = {label: set() for label in LABELS}
    for record in records:
        for entity in record["entities"]:
            surface = record["text"][entity["start"] : entity["end"]]
            if surface.strip():
                surfaces[entity["label"]].add(surface)
    return {label: sorted(values) for label, values in surfaces.items()}


def _surface_vectors(
    surfaces: list[str],
    tokenizer: Any,
    encoder: torch.nn.Module,
    *,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Represent each entity surface by its mean input-embedding vector."""

    if not surfaces:
        return {}
    embedding = encoder.get_input_embeddings()
    weight = embedding.weight.detach()
    vectors: list[torch.Tensor] = []
    encoder.eval()
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
                encoded["attention_mask"],
                device=weight.device,
                dtype=weight.dtype,
            )
            token_vectors = weight[input_ids].float()
            pooled = (token_vectors * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled /= attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            vectors.extend(functional.normalize(pooled, dim=1).cpu())
    return dict(zip(surfaces, vectors, strict=True))


def _kmeans(vectors: torch.Tensor, cluster_count: int, seed: int) -> list[int]:
    """Cluster normalized vectors with deterministic cosine K-Means."""

    if len(vectors) == 0:
        return []
    cluster_count = min(cluster_count, len(vectors))
    if cluster_count == 1:
        return [0] * len(vectors)

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(len(vectors), generator=generator)[:cluster_count]
    centroids = vectors[initial].clone()
    for _ in range(25):
        assignments = (vectors @ centroids.T).argmax(dim=1)
        updated: list[torch.Tensor] = []
        for cluster_index in range(cluster_count):
            members = vectors[assignments == cluster_index]
            centroid = members.mean(dim=0) if len(members) else centroids[cluster_index]
            updated.append(functional.normalize(centroid.unsqueeze(0), dim=1).squeeze(0))
        new_centroids = torch.stack(updated)
        new_assignments = (vectors @ new_centroids.T).argmax(dim=1)
        centroids = new_centroids
        if torch.equal(assignments, new_assignments):
            break
    return (vectors @ centroids.T).argmax(dim=1).tolist()


def _make_augmented_record(
    record: JsonObject,
    replacements: dict[int, str],
    variant_index: int,
) -> JsonObject:
    """Apply selected replacements and preserve every original entity with new offsets."""

    parts: list[str] = []
    entities: list[JsonObject] = []
    source_cursor = 0
    output_cursor = 0
    for entity_index, entity in enumerate(record["entities"]):
        start, end = entity["start"], entity["end"]
        prefix = record["text"][source_cursor:start]
        parts.append(prefix)
        output_cursor += len(prefix)
        surface = replacements.get(entity_index, record["text"][start:end])
        entity_start = output_cursor
        parts.append(surface)
        output_cursor += len(surface)
        entities.append({"label": entity["label"], "start": entity_start, "end": output_cursor})
        source_cursor = end
    parts.append(record["text"][source_cursor:])
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
    encoder: torch.nn.Module,
    *,
    enabled: bool,
    probability: float,
    num_clusters: int,
    max_candidates: int,
    seed: int,
) -> tuple[list[JsonObject], dict[str, Any]]:
    """Append one same-label cluster-based variant for selected train records."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation.probability must be in [0, 1]")
    if num_clusters < 1:
        raise ValueError("augmentation.num_clusters must be positive")
    if max_candidates < 1:
        raise ValueError("augmentation.max_candidates must be positive")
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
        stats["output_records"] = len(records)
        return list(records), stats

    surfaces_by_label = _entity_surfaces(records)
    all_surfaces = [surface for surfaces in surfaces_by_label.values() for surface in surfaces]
    vectors = _surface_vectors(all_surfaces, tokenizer, encoder)
    clusters: dict[tuple[str, int], list[str]] = defaultdict(list)
    assignments: dict[tuple[str, str], int] = {}
    for label_index, label in enumerate(LABELS):
        surfaces = surfaces_by_label[label]
        if not surfaces:
            continue
        cluster_count = min(num_clusters, max(1, len(surfaces) // 2))
        matrix = torch.stack([vectors[surface] for surface in surfaces])
        cluster_ids = _kmeans(matrix, cluster_count, seed + label_index)
        for surface, cluster_id in zip(surfaces, cluster_ids, strict=True):
            clusters[(label, cluster_id)].append(surface)
            assignments[(label, surface)] = cluster_id

    randomizer = random.Random(seed)
    augmented = list(records)
    variant_index = 0
    for record in records:
        if randomizer.random() >= probability:
            continue
        replacements: dict[int, str] = {}
        for entity_index, entity in enumerate(record["entities"]):
            source = record["text"][entity["start"] : entity["end"]]
            cluster_id = assignments.get((entity["label"], source))
            candidates = [
                candidate
                for candidate in clusters.get((entity["label"], cluster_id), [])
                if candidate != source
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda candidate: float(vectors[source] @ vectors[candidate]),
                reverse=True,
            )
            replacements[entity_index] = randomizer.choice(candidates[:max_candidates])
        if not replacements:
            continue
        variant_index += 1
        new_record = _make_augmented_record(record, replacements, variant_index)
        if len(new_record["entities"]) != len(record["entities"]):
            raise AssertionError("cluster augmentation lost an entity")
        augmented.append(new_record)
        stats["added_records"] += 1
        stats["replaced_entities"] += len(replacements)

    stats["output_records"] = len(augmented)
    stats["cluster_count"] = len(clusters)
    return augmented, stats
