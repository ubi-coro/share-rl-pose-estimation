Open problems
=============

Near-term questions
-------------------

- How far should scripted open-loop primitives go before they become a separate
  trajectory language?
- What is the cleanest way to support mixed policy + scripted residual control
  without bloating the primitive/env API?
- How should relative-frame semantics be documented and surfaced so users can
  predict target resolution more confidently?

Codebase maintenance themes
---------------------------

- Keep import surfaces light. Circular imports are still easy to trigger in the
  env stack.
- Keep examples and docs in sync with the current primitive lifecycle.
- Keep transition logic narrow. If a new behavior needs substantial runtime
  machinery, it probably belongs in the primitive env or config entry hook
  rather than inside a transition class.

Good contributions
------------------

- more end-to-end examples for joint-only robots that rely on FK-added EE pose
- clearer docs around task-frame origin semantics
- lightweight visualization/debugging for primitive targets and transitions
