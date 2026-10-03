"""Install placeholder ``torch`` and ``torchvision`` modules so that ``common.py`` can be imported
for its grouping code on a machine without the GPU stack (the manuscript renderer and the
verification script only need ``common.build_groups``). Import this module before ``common``.
If the real packages are installed, this module does nothing.
"""
import sys
import types

try:  # real packages present: nothing to do
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:
    class _Base:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

    class _Permissive(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            return _Base

    def _mod(name, **attrs):
        m = _Permissive(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    _torch = _mod('torch', cuda=types.SimpleNamespace(is_available=lambda: False))
    _torch.nn = _mod('torch.nn', Module=_Base)
    _mod('torch.nn.functional')
    _torch.utils = _mod('torch.utils', data=_mod('torch.utils.data'))
    _tv = _mod('torchvision')
    _tv.models = _mod('torchvision.models')
    _tv.transforms = _mod('torchvision.transforms')
    _mod('torchvision.models.efficientnet')
