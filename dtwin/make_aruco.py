from __future__ import annotations

import pathlib

import yaml

from dtwin.board import BoardConfig, generate_board_image


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    board_cfg = BoardConfig(
        dictionary=cfg["board"]["dictionary"],
        markers_x=cfg["board"]["markers_x"],
        markers_y=cfg["board"]["markers_y"],
        marker_length_m=cfg["board"]["marker_length_m"],
        marker_separation_m=cfg["board"]["marker_separation_m"],
        axis_length_m=cfg["board"]["axis_length_m"],
    )

    out_path = ROOT / cfg["board"]["board_image_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generate_board_image(
        cfg=board_cfg,
        out_path=str(out_path),
        pixels_per_meter=cfg["board"]["pixels_per_meter"],
        margin_px=cfg["board"]["margin_px"],
    )

    print(f"Saved board image to: {out_path}")


if __name__ == "__main__":
    main()