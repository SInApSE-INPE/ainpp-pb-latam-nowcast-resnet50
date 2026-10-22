from pathlib import Path

from resnet50_nowcast._runtime.deployment import render_units

if __name__ == "__main__":
    render_units(
        Path(__file__).resolve().parent.parent, "resnet50_nowcast", "ResNet50", "resnet50-now"
    )
