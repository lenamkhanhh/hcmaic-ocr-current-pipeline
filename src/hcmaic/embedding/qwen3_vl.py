"""Local-files-only Qwen3-VL-Embedding-2B query provider.

The catalog embeddings were produced with the official Qwen3-VL embedding
architecture pinned in the merge manifest.  This adapter mirrors that image
pooling contract and additionally exposes the text tower for TKIS queries.
It never downloads weights unless the caller explicitly opts into network
access; the local retrieval CLI defaults to ``local_files_only=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"


def _last_token_pool(hidden: Any, attention_mask: Any, torch: Any, functional: Any) -> Any:
    if hidden.ndim != 3 or attention_mask.ndim != 2 or hidden.shape[:2] != attention_mask.shape:
        raise RuntimeError(
            f"Qwen hidden/mask shapes are incompatible: {tuple(hidden.shape)} / "
            f"{tuple(attention_mask.shape)}"
        )
    # Right-padding is configured on the processor.  This expression works
    # for both a batch of different lengths and a fully padded batch.
    last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    row = torch.arange(hidden.shape[0], device=hidden.device)
    return functional.normalize(hidden[row, last], p=2, dim=-1)


class Qwen3VLEmbeddingProvider(EmbeddingProvider):
    """Official Qwen3-VL image/text encoder used for the 2048-D index."""

    name = "qwen3-vl-embedding-2b"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        revision: str = DEFAULT_REVISION,
        local_files_only: bool = True,
        model_path: Path | None = None,
        batch_size: int = 2,
        max_length: int = 8192,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be >= 1")
        try:
            import torch
            from transformers import Qwen3VLModel, Qwen3VLPreTrainedModel, Qwen3VLProcessor
            from transformers.utils import ModelOutput
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Qwen3-VL requires the pinned transformers/torch runtime; install "
                "the offline embedding extras before serving"
            ) from exc

        @dataclass
        class EmbeddingOutput(ModelOutput):
            last_hidden_state: Any = None
            attention_mask: Any = None

        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            _checkpoint_conversion_mapping: dict[str, str] = {}
            accepts_loss_kwargs = False

            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

            def forward(self, attention_mask: Any = None, **kwargs: Any) -> Any:
                outputs = self.model(attention_mask=attention_mask, **kwargs)
                return EmbeddingOutput(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=attention_mask,
                )

        self._torch = torch
        self._functional = __import__("torch.nn.functional", fromlist=["normalize"])
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._revision = revision
        self._local_files_only = local_files_only
        self._batch = batch_size
        self._max_length = max_length
        source = str(model_path) if model_path is not None else model_name
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": False,
            "output_loading_info": True,
            "local_files_only": local_files_only,
        }
        if model_path is None:
            load_kwargs["revision"] = revision
        try:
            model, loading_info = Qwen3VLForEmbedding.from_pretrained(source, **load_kwargs)
            processor_kwargs: dict[str, Any] = {
                "padding_side": "right",
                "local_files_only": local_files_only,
            }
            if model_path is None:
                processor_kwargs["revision"] = revision
            processor = Qwen3VLProcessor.from_pretrained(source, **processor_kwargs)
        except (OSError, ValueError) as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise RuntimeError(
                f"Qwen3-VL model {model_name!r} is unavailable from {mode}; "
                "cache the exact revision first or explicitly allow network access"
            ) from exc
        self._model = model.to(self._device).eval()
        self._processor = processor
        self._dtype = dtype
        self._loading_info = loading_info
        self._parameter_count = sum(parameter.numel() for parameter in self._model.parameters())
        self._dimension = self._read_dimension(self._model.config)
        self.version = f"qwen3-vl-embedding-2b:{model_name}@{revision}"

    @staticmethod
    def _read_dimension(config: Any) -> int:
        for candidate in (
            config,
            getattr(config, "text_config", None),
            getattr(config, "vision_config", None),
        ):
            if candidate is None:
                continue
            for field in ("hidden_size", "projection_dim", "projection_size"):
                value = getattr(candidate, field, None)
                if value is not None:
                    return int(value)
        raise RuntimeError("Qwen3-VL model config has no embedding dimension")

    @property
    def dimension(self) -> int:
        return self._dimension

    def info(self) -> dict[str, Any]:
        data = super().info()
        data.update(
            {
                "model_name": self._model_name,
                "model_revision": self._revision,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch,
                "max_length": self._max_length,
                "dtype": str(self._dtype),
                "parameter_count": self._parameter_count,
                "local_files_only": self._local_files_only,
                "loader": "Qwen3VLForEmbedding official architecture",
                "evidence_level": "REAL_PROVIDER",
            }
        )
        return data

    def _encode(self, inputs: Any) -> np.ndarray:
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state
        attention = outputs.attention_mask
        embeddings = _last_token_pool(hidden, attention, self._torch, self._functional)
        array = embeddings.detach().float().cpu().numpy().astype(np.float32)
        if array.ndim != 2 or array.shape[1] != self._dimension:
            raise RuntimeError(
                f"Qwen3-VL returned feature shape {array.shape}; expected (*, {self._dimension})"
            )
        normalized = l2_normalize(array)
        if not np.isfinite(normalized).all():
            raise RuntimeError("Qwen3-VL returned non-finite embeddings")
        return normalized

    def _conversations(
        self,
        *,
        texts: list[str] | None = None,
        paths: list[Path] | None = None,
    ) -> list[list[dict[str, Any]]]:
        if (texts is None) == (paths is None):
            raise ValueError("provide exactly one of texts or paths")
        if texts is not None:
            return [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Represent the user's input."}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": text}]},
                ]
                for text in texts
            ]
        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Represent the user's input."}],
                },
                {"role": "user", "content": [{"type": "image", "image": path.resolve().as_uri()}]},
            ]
            for path in paths or []
        ]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            conversations = self._conversations(texts=batch)
            rendered = self._processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=rendered,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            outputs.append(self._encode(inputs))
        return np.vstack(outputs).astype(np.float32)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dimension), dtype=np.float32)
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Qwen image queries require qwen-vl-utils in the serving environment"
            ) from exc
        outputs: list[np.ndarray] = []
        for start in range(0, len(paths), self._batch):
            batch_paths = [Path(path).resolve() for path in paths[start : start + self._batch]]
            conversations = self._conversations(paths=batch_paths)
            rendered = self._processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False
            )
            images, videos, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            video_metadata_for_processor: Any = None
            videos_for_processor: Any = videos
            if videos:
                video_pairs = list(videos)
                videos_for_processor = [pair[0] for pair in video_pairs]
                video_metadata_for_processor = [pair[1] for pair in video_pairs]
            inputs = self._processor(
                text=rendered,
                images=images,
                videos=videos_for_processor,
                video_metadata=video_metadata_for_processor,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            ).to(self._device)
            outputs.append(self._encode(inputs))
        return np.vstack(outputs).astype(np.float32)
