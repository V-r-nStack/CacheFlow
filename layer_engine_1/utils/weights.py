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
    """
    Download checkpoint file to a temporary location.
    Uses streaming so large files do not fully load into RAM.
    """

    response = requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    response.raise_for_status()

    with destination.open("wb") as handle:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                handle.write(chunk)


def _load_checkpoint_as_state_dict(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """
    Load either .bin or .safetensors checkpoint
    into a PyTorch state dict.
    """

    if checkpoint_path.suffix == ".safetensors":

        try:
            from safetensors.torch import load_file

        except ImportError as exc:

            raise RuntimeError(
                "Loading .safetensors checkpoints requires "
                "the optional 'safetensors' package."
            ) from exc

        return load_file(str(checkpoint_path))

    loaded_object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(loaded_object, dict):

        raise TypeError(
            "Expected a PyTorch state dictionary "
            "at the checkpoint path."
        )

    return loaded_object


def download_gpt2_weights(
    save_path: str | os.PathLike[str] | None = None,
) -> Path:
    """
    Download GPT-2 checkpoint weights and save them
    locally as a normalized PyTorch state dict.
    """

    destination = (
        Path(save_path)
        if save_path is not None
        else (
            Path.cwd()
            / "weights"
            / "gpt2_124m_state_dict.pt"
        )
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_error: Exception | None = None

    for url in GPT2_CHECKPOINT_URLS:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(url).suffix,
        ) as tmp_file:

            temp_path = Path(tmp_file.name)

        try:

            print(f"\nDownloading:\n{url}\n")

            _download_to_path(url, temp_path)

            # Quick sanity check for incomplete downloads
            file_size_mb = (
                temp_path.stat().st_size
                / 1024
                / 1024
            )

            print(
                f"Downloaded size: "
                f"{file_size_mb:.2f} MB"
            )

            if file_size_mb < 100:
                raise RuntimeError(
                    "Downloaded checkpoint is too small. "
                    "Download likely failed."
                )

            print("\nLoading checkpoint...\n")

            state_dict = _load_checkpoint_as_state_dict(
                temp_path
            )

            print(
                f"Loaded {len(state_dict)} tensors."
            )

            print(
                f"\nSaving checkpoint to:\n"
                f"{destination}\n"
            )

            torch.save(state_dict, destination)

            print("Checkpoint saved successfully.")

            return destination

        except Exception as exc:

            last_error = exc

            print("\nDownload failed.\n")
            print(type(exc).__name__)
            print(exc)

        finally:

            temp_path.unlink(missing_ok=True)

    assert last_error is not None

    raise RuntimeError(
        "Failed to download GPT-2 weights "
        "from HuggingFace."
    ) from last_error


def main():

    path = download_gpt2_weights()

    print("\n================================")
    print("GPT-2 weights ready.")
    print(path)
    print("================================")


if __name__ == "__main__":
    main()