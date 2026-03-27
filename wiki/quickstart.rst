Quickstart
==========

Core directories
----------------

- ``src/share/envs/manipulation_primitive``
  Primitive configs, task-frame math, env runtime, and processor wiring.
- ``src/share/envs/manipulation_primitive_net``
  MP-Net config, runtime orchestration, and transition logic.
- ``src/share/workspace/mpnet.py``
  Serialization and editing helpers for config-driven MP-Nets.
- ``examples/``
  Runnable demonstrations for real and mock setups.

Lifecycle in one sentence
-------------------------

MP-Nets step the current primitive, evaluate transitions, and when a primitive
switch fires the caller is expected to call ``reset()`` so the next primitive
can enter using the stored boundary context.

Key design rules
----------------

- Configs describe behavior but do not keep runtime state.
- Primitive envs own runtime target state.
- Entry-time target resolution happens in config ``on_entry(...)`` hooks.
- If a primitive needs different stepping behavior, it should provide its own
  env subclass in ``make(...)``.

Suggested reading order
-----------------------

1. ``src/share/envs/ENVS.md``
2. ``src/share/envs/manipulation_primitive/config_manipulation_primitive.py``
3. ``src/share/envs/manipulation_primitive/env_manipulation_primitive.py``
4. ``src/share/envs/manipulation_primitive_net/env_manipulation_primitive_net.py``
5. ``examples/demo_mp_net_real.py`` and the mock MP-Net examples
