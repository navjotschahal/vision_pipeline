"""In-memory nearest-neighbour retrieval over frozen DINOv2 image features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from vision_pipeline.perception.contracts import FeatureBatch


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    item_id: str
    cosine_similarity: float

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must be non-empty")
        if not math.isfinite(self.cosine_similarity):
            raise ValueError("cosine_similarity must be finite")


class DinoFeatureIndex:
    """A small exact index; production callers can replace it with FAISS or a DB."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._model_id: str | None = None
        self._item_ids: list[str] = []
        self._descriptors: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._item_ids)

    @staticmethod
    def _descriptor(batch: FeatureBatch[torch.Tensor]) -> torch.Tensor:
        descriptor = batch.global_features
        if descriptor.ndim != 2 or descriptor.shape[0] != 1:
            raise ValueError("global features must have single-image [1, embedding] shape")
        return functional.normalize(descriptor[0].detach().to(dtype=torch.float32), dim=0)

    def add(self, item_id: str, batch: FeatureBatch[torch.Tensor]) -> None:
        if not item_id.strip():
            raise ValueError("item_id must be non-empty")
        descriptor = self._descriptor(batch)
        if self._model_id is None:
            self._model_id = batch.model_id
        elif batch.model_id != self._model_id:
            raise ValueError("all indexed descriptors must use the same model")
        if self._descriptors and descriptor.shape != self._descriptors[0].shape:
            raise ValueError("indexed descriptor dimensions differ")
        if self._descriptors and descriptor.device != self._descriptors[0].device:
            raise ValueError("all indexed descriptors must share a device")

        if item_id in self._item_ids:
            index = self._item_ids.index(item_id)
            self._descriptors[index] = descriptor
            return
        self._item_ids.append(item_id)
        self._descriptors.append(descriptor)
        if len(self._item_ids) > self.capacity:
            self._item_ids.pop(0)
            self._descriptors.pop(0)

    def query(
        self, batch: FeatureBatch[torch.Tensor], top_k: int = 1
    ) -> tuple[RetrievalMatch, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._descriptors:
            return ()
        if batch.model_id != self._model_id:
            raise ValueError("query and index must use the same model")
        query = self._descriptor(batch)
        database = torch.stack(self._descriptors)
        if query.device != database.device:
            raise ValueError("query and index descriptors must share a device")
        similarities = database @ query
        count = min(top_k, len(self._item_ids))
        values, indices = torch.topk(similarities, k=count, largest=True, sorted=True)
        return tuple(
            RetrievalMatch(
                item_id=self._item_ids[int(index.item())],
                cosine_similarity=float(value.item()),
            )
            for value, index in zip(values, indices, strict=True)
        )


__all__ = ["DinoFeatureIndex", "RetrievalMatch"]
