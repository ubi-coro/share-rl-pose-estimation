# share-rl

`share-rl` is a LeRobot v5 extension focused on structured manipulation workflows.

It packages:

- manipulation primitive environments and processor pipelines with task-frame constraints
- manipulation primitive nets (MP-Nets) for graph-structured high-level behavior
- an interactive workspace CLI that acts as a bounded coding/programming agent for MP-Nets
- experimental offline RL policy code under `src/share/policies`
- hardware integrations for UR and ViperX-style robots, plus LeRobot-compatible teleoperation hooks

## Status

This repository is installable and documented for release-oriented use, but not every subsystem is equally mature.

- Ready to package and install: the `share` Python package, MP-Net/env code, workspace CLI, and the standalone robot packages `lerobot_robot_ur`, `lerobot_robot_urV2`, and `lerobot_robot_viper`
- Still experimental: most code under `src/share/workspace` and `src/share/policies`
- Placeholder-only today: the namespace folders under `src/share/teleoperators` do not yet contain standalone teleoperator implementations in this repo

## Installation

Python 3.12 is a good default for this repo.

```bash
conda create -n share python=3.12
conda activate share
pip install -e .
```

The package pins LeRobot to the v5 line and supports optional extras:

```bash
pip install -e ".[ur]"
pip install -e ".[spacemouse]"
pip install -e ".[aloha]"
pip install -e ".[all]"
```

What each extra is for:

- `ur`: UR RTDE dependencies and low-level synchronization helpers
- `spacemouse`: SpaceMouse teleoperation dependency expected by the LeRobot-style stack
- `aloha`: Dynamixel/serial dependencies commonly needed for Aloha or ViperX-style hardware
- `all`: union of `ur`, `spacemouse`, and `aloha`

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Installed Commands

Editable install exposes a few convenient entry points:

- `share-record`
- `share-train`
- `share-eval`
- `share-workspace`

The workspace shell is the interactive interface for building and inspecting MP-Nets:

```bash
share-workspace --workspace ./workspace --project demo --task pick-place
```

## Repository Layout

- `src/share/envs`: manipulation primitive envs, configs, transitions, processor steps
- `src/share/workspace`: bounded interactive agent/runtime for MP-Net authoring
- `src/share/policies`: offline RL and value/policy modeling experiments
- `src/share/robots`: UR and Viper robot packages
- `src/share/scripts`: recording, training, evaluation, and workspace CLIs
- `tests`: focused unit tests and smoke coverage

## Notes

- The codebase assumes LeRobot v5 APIs.
- If you manage PyTorch/CUDA manually, install the compatible torch build for your machine before running training-heavy workflows.
- `share-workspace --voice` is optional and still requires `SpeechRecognition` plus working microphone/audio support; it is not bundled into the default install.
