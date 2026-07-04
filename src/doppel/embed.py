"""CLIP image embeddings. The heavy imports (torch, open_clip) are lazy so
the rest of the app never pays for them; tests use a fake Embedder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

BATCH_SIZE = 32  # per SPEC "Stage 3 — similar"


class Embedder(Protocol):
    """Maps image files to L2-normalized embedding vectors."""

    def embed(self, paths: list[Path]) -> np.ndarray:
        """(len(paths), 512) float32, rows L2-normalized."""
        ...


def pick_device(torch_module: Any) -> str:
    """MPS when available, CPU fallback (per SPEC)."""
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbedder:
    """open_clip ViT model; loaded lazily on first embed call."""

    def __init__(self, model_spec: str) -> None:
        import threading

        # model_spec e.g. "ViT-B-32/laion2b_s34b_b79k" from config.toml
        self._model_name, self._pretrained = model_spec.split("/", 1)
        self._loaded: tuple[Any, Any, str] | None = None
        # embed() is called only from the similar stage's single consumer
        # thread, but guard the one-time load anyway so a future concurrent
        # caller can't race two model loads onto the GPU.
        self._load_lock = threading.Lock()

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            with self._load_lock:
                if self._loaded is None:
                    import open_clip
                    import torch

                    device = pick_device(torch)
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self._model_name, pretrained=self._pretrained
                    )
                    model = model.to(device).eval()
                    self._loaded = (model, preprocess, device)
        return self._loaded

    def embed(self, paths: list[Path]) -> np.ndarray:
        import torch

        model, preprocess, device = self._load()
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in paths]
        ).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
