from __future__ import annotations

import logging
import sys
from pathlib import Path


def _ensure_tools_on_path() -> None:
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))


def main() -> int:
    _ensure_tools_on_path()
    from hand_eye_calibration import main as calibration_main

    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "manual"])

    logging.getLogger(__name__).warning(
        "save_robot_poses.py is deprecated. Use hand_eye_calibration.py --mode manual instead."
    )
    return calibration_main()


if __name__ == "__main__":
    sys.exit(main())
