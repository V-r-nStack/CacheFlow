from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests
import torch

GPT2_CHECKPOINT_URLS: tuple[str, ...] = (
    "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
    "https://huggingface.co/gpt2/resolve/main/model.safetensors",
)


def _download_to_path(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _load_checkpoint_as_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Loading .safetensors checkpoints requires the optional 'safetensors' package. "
                "Install it or download the .bin checkpoint instead."
            ) from exc

        return load_file(str(checkpoint_path))

    loaded_object = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded_object, dict):
        raise TypeError("Expected a PyTorch state dictionary at the checkpoint path.")
    return loaded_object


def download_gpt2_weights(save_path: str | os.PathLike[str] | None = None) -> Path:
    destination = (
        Path(save_path)
        if save_path is not None
        else Path(__file__).resolve().parents[1] / "weights" / "gpt2_124m_state_dict.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in GPT2_CHECKPOINT_URLS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(url).suffix) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            _download_to_path(url, temp_path)
            state_dict = _load_checkpoint_as_state_dict(temp_path)
            torch.save(state_dict, destination)
            return destination
        except Exception as exc:  # pragma: no cover - network and optional-format failure path
            last_error = exc
        finally:
            temp_path.unlink(missing_ok=True)

    assert last_error is not None
    raise RuntimeError("Failed to download GPT-2 weights from HuggingFace.") from last_error
