# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Facilities for pickling Python code alongside other data.

The pickled code is automatically imported into a separate Python module
during unpickling. This way, any previously exported pickles will remain
usable even if the original code is no longer available, or if the current
version of the code is not consistent with what was originally pickled."""

import sys
import pickle
import io
import inspect
import copy
import uuid
import types
import dnnlib

#----------------------------------------------------------------------------

_version            = 6         # internal version number
_decorators         = set()     # {decorator_class, ...}
_import_hooks       = []        # [hook_function, ...]
_module_to_src_dict = dict()    # {module: src, ...}
_src_to_module_dict = dict()    # {src: module, ...}

#----------------------------------------------------------------------------

def persistent_class(orig_class):
    r"""Class decorator that extends a given class to save its source code
    when pickled.

    Example:

        from torch_utils import persistence

        @persistence.persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
                ...

        @persistence.persistent_class
        class MyLayer(torch.nn.Module):
            ...

    When pickled, any instance of `MyNetwork` and `MyLayer` will save its
    source code alongside other internal state (e.g., parameters, buffers,
    and submodules). This way, any previously exported pickle will remain
    usable even if the class definitions have been modified or are no
    longer available.

    The decorator saves the source code of the entire Python module
    containing the decorated class. It does *not* save the source code of
    any imported modules. Thus, the imported modules must be available
    during unpickling, also including `torch_utils.persistence` itself.

    It is ok to call functions defined in the same module from the
    decorated class. However, if the decorated class depends on other
    classes defined in the same module, they must be decorated as well.
    This is illustrated in the above example in the case of `MyLayer`.

    It is also possible to employ the decorator just-in-time before
    calling the constructor. For example:

        cls = MyLayer
        if want_to_make_it_persistent:
            cls = persistence.persistent_class(cls)
        layer = cls(num_inputs, num_outputs)

    As an additional feature, the decorator also keeps track of the
    arguments that were used to construct each instance of the decorated
    class. The arguments can be queried via `obj.init_args` and
    `obj.init_kwargs`, and they are automatically pickled alongside other
    object state. A typical use case is to first unpickle a previous
    instance of a persistent class, and then upgrade it to use the latest
    version of the source code:

        with open('old_pickle.pkl', 'rb') as f:
            old_net = pickle.load(f)
        new_net = MyNetwork(*old_obj.init_args, **old_obj.init_kwargs)
        misc.copy_params_and_buffers(old_net, new_net, require_all=True)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(type='class', version=_version, module_src=self._orig_module_src, class_name=self._orig_class_name, state=fields[2])
                fields[0] = _reconstruct_persistent_obj # reconstruct func
                fields[1] = (meta,) # reconstruct args
                fields[2] = None # state dict
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    _decorators.add(Decorator)
    return Decorator

#----------------------------------------------------------------------------

def is_persistent(obj):
    r"""Test whether the given object or class is persistent, i.e.,
    whether it will save its source code when pickled.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators # pylint: disable=unidiomatic-typecheck

#----------------------------------------------------------------------------

def import_hook(hook):
    r"""Register an import hook that is called whenever a persistent object
    is being unpickled. A typical use case is to patch the pickled source
    code to avoid errors and inconsistencies when the API of some imported
    module has changed.

    The hook should have the following signature:

        hook(meta) -> modified meta

    `meta` is an instance of `dnnlib.EasyDict` with the following fields:

        type:       Type of the persistent object, e.g. `'class'`.
        version:    Internal version number of `torch_utils.persistence`.
        module_src  Original source code of the Python module.
        class_name: Class name in the original Python module.
        state:      Internal state of the object.

    Example:

        @persistence.import_hook
        def wreck_my_network(meta):
            if meta.class_name == 'MyNetwork':
                print('MyNetwork is being imported. I will wreck it!')
                meta.module_src = meta.module_src.replace("True", "False")
            return meta
    """
    assert callable(hook)
    _import_hooks.append(hook)

#----------------------------------------------------------------------------

def _reconstruct_persistent_obj(meta):
    r"""Hook that is called internally by the `pickle` module to unpickle
    a persistent object.
    """
    meta = dnnlib.EasyDict(meta)
    meta.state = dnnlib.EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == 'class'
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, '__setstate__', None)
    if callable(setstate):
        setstate(meta.state) # pylint: disable=not-callable
    else:
        obj.__dict__.update(meta.state)
    return obj

#----------------------------------------------------------------------------

def _module_to_src(module):
    r"""Query the source code of a given Python module.
    """
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src

def _src_to_module(src):
    r"""Get or create a Python module for the given source code.
    """
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__) # pylint: disable=exec-used
    return module

#----------------------------------------------------------------------------

def _check_pickleable(obj):
    r"""Check that the given object is pickleable, raising an exception if
    it is not. This function is expected to be considerably more efficient
    than actually pickling the object.
    """
    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None # Python primitive types are pickleable.
        if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor', 'torch.nn.parameter.Parameter']:
            return None # NumPy arrays and PyTorch tensors are pickleable.
        if is_persistent(obj):
            return None # Persistent objects are pickleable, by virtue of the constructor check.
        return obj
    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)

#----------------------------------------------------------------------------
