#!/usr/bin/env python3
"""Small CLI for testing the SAM3 ZeroMQ client on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from share.pose_estimation.sam3_client import Sam3Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("prompt", help="Text prompt to send to SAM3.")
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5565",
        help="ZeroMQ endpoint for the SAM3 publisher.",
    )
    parser.add_argument(
        "--image-format",
        default="bgr",
        choices=("bgr", "rgb"),
        help="Pixel format expected by the SAM3 service.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional confidence threshold override for this request.",
    )
    parser.add_argument(
        "--mask-out",
        type=Path,
        default=None,
        help="Optional output path for the predicted mask image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")

    with Sam3Client(endpoint=args.endpoint) as client:
        result = client.segment_image(
            image=image,
            image_format=args.image_format,
            prompt=args.prompt,
            confidence_threshold=args.confidence_threshold,
        )

    mask = result["mask"]
    box = result.get("box")
    score = result.get("score")

    print(f"prompt: {args.prompt}")
    print(f"mask shape: {mask.shape}")
    print(f"mask pixels: {int(mask.astype(bool).sum())}")
    print(f"score: {score}")
    print(f"box: {box.tolist() if box is not None else None}")

    if args.mask_out is not None:
        mask_image = (mask > 0).astype("uint8") * 255
        cv2.imwrite(str(args.mask_out), mask_image)
        print(f"saved mask to: {args.mask_out}")


if __name__ == "__main__":
    main()
