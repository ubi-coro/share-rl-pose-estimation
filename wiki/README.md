# `share-rl` Wiki

This folder is a small top-level documentation scaffold intended to grow into a structured docs site.

It is written with Sphinx-style documentation in mind:

- `conf.py` contains a minimal Sphinx configuration
- `index.rst` is the root landing page
- the remaining `.rst` pages are organized like a small handbook

If `sphinx-build` is available, you can build the docs from this folder:

```bash
cd wiki
make html
```

The initial page set focuses on the current MP-Net architecture, examples, and open questions rather than exhaustive API reference.
