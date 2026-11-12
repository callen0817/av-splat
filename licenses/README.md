# Third-party components

- SPZ: official Niantic implementation at commit `855173912dee2a48cb539657f46c4ade775f48c9`, MIT. Installed through requirements-spz.txt; no independent file-format imitation is used.
- Optional gsplat backend: source commit `512d366b67073d77ca099ede742683c165dfc23b`, Apache 2.0. No pretrained weights or training datasets are downloaded. CUDA execution has not been validated in the CPU-only cloud task.
- Both new desktop applications use dynamically linked PySide6 bindings and Qt modules. Preserve their LGPL/commercial licence notices and distribution obligations. Package recipes retain installed PySide6/Shiboken licence metadata. AV Snap retains its pre-existing PyQt5 stack.

The exact dependency versions installed in this task are recorded by environment validation; licence summaries are not a substitute for distribution review of the final platform package.
