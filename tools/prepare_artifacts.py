"""Extract inference-only ResNet50 weights and a checksummed artifact manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from resnet50_nowcast._runtime.artifacts import sha256


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--bundle", type=Path, required=True)
    args = cli.parse_args()
    args.bundle.mkdir(parents=True, exist_ok=True)
    target = args.bundle / "resnet50.pth"
    manifest_path = args.bundle / "manifest.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Use a new bundle directory; existing artifacts are preserved")
    (mean,) = args.bundle.glob("*_log_mean_*.npy")
    (std,) = args.bundle.glob("*_log_std_*.npy")
    # Only scalar NumPy metadata in the supplied training checkpoint needs allowlisting.
    allowed = [
        (np._core.multiarray.scalar, "numpy._core.multiarray.scalar"),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model_state_dict"]
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in cleaned or not torch.is_tensor(value):
            raise ValueError("Invalid or ambiguous state dictionary")
        cleaned[key] = value
    input_steps = cleaned["backbone.conv1.weight"].shape[1]
    output_steps = cleaned["outc.weight"].shape[0]
    if input_steps < 1 or output_steps < 1:
        raise ValueError("Invalid checkpoint sequence lengths")
    torch.save(cleaned, target)
    manifest = {
        "schema_version": 1,
        "version": args.bundle.name,
        "model": "ResNet50",
        "architecture": {
            "encoder": "torchvision_resnet50",
            "input_steps": input_steps,
            "output_steps": output_steps,
            "output_activation": "relu_normalized",
        },
        "reference_input_product": "gsmap_nrt",
        "training_latitude_order": None,
        "training_domain_verified": False,
        "source_checkpoint": {"name": args.checkpoint.name, "sha256": sha256(args.checkpoint)},
        "training_args": checkpoint.get("args", {}),
        "files": {
            role: {"name": path.name, "sha256": sha256(path)}
            for role, path in (("checkpoint", target), ("mean", mean), ("std", std))
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Created {target}: {input_steps} input hours, {output_steps} direct output hours")


if __name__ == "__main__":
    main()
