"""Bounded reader for this project's plain ZIP tensor state dictionary.

No torch import and no generic pickle globals. Tensor storage stays lazy until
prefix selection. This is a file boundary, separate from every numerical kernel.
"""

from collections import OrderedDict, namedtuple
import io
import pickle
import pickletools
import zipfile

import numpy as np


_Storage = namedtuple("Storage", "dtype key count")
_Tensor = namedtuple("Tensor", "storage offset shape strides")


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _rebuild(storage, offset, shape, strides, requires_grad, hooks, metadata=None):
    if not isinstance(storage, _Storage) or not _integer(offset) or offset < 0:
        raise ValueError("invalid tensor storage or offset")
    if not isinstance(shape, tuple) or not isinstance(strides, tuple) or len(shape) != len(strides) or len(shape) > 8:
        raise ValueError("invalid tensor shape/strides")
    if any(not _integer(n) or n <= 0 for n in shape) or any(not _integer(s) or s < 0 for s in strides):
        raise ValueError("only positive dimensions and nonnegative strides are supported")
    if type(requires_grad) is not bool or hooks not in (None, {}) or metadata is not None:
        raise ValueError("only plain tensor state dictionaries are supported")
    if offset + sum((n-1)*s for n, s in zip(shape, strides)) >= storage.count:
        raise ValueError("tensor view exceeds storage bounds")
    return _Tensor(storage, offset, shape, strides)


class _Restricted(pickle.Unpickler):
    def __init__(self, payload, max_bytes):
        super().__init__(io.BytesIO(payload))
        self.max_bytes = max_bytes

    def find_class(self, module, name):
        allowed = {("collections", "OrderedDict"): OrderedDict,
                   ("torch", "FloatStorage"): "f4", ("torch", "BoolStorage"): "?",
                   ("torch._utils", "_rebuild_tensor_v2"): _rebuild}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5:
            raise ValueError("invalid storage reference")
        tag, dtype, key, location, count = value
        if tag != "storage" or dtype not in ("f4", "?") or not isinstance(key, str) or not key.isascii() or not key.isdecimal():
            raise ValueError("unsupported storage reference")
        if not isinstance(location, str) or not _integer(count) or count <= 0 or count*np.dtype(dtype).itemsize > self.max_bytes:
            raise ValueError("storage exceeds allowed size")
        return _Storage(dtype, key, count)


def load_checkpoint(path, prefix=None, *, max_bytes=128*1024**2):
    """Return float32/bool raw arrays. An explicit prefix is filtered and stripped.

    Only the observed FloatStorage/BoolStorage plain-tensor format is accepted.
    Every descriptor is bounds-checked, including unselected tensors. Each
    selected output is an independent copy, with an aggregate allocation limit.
    """
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("prefix must be a string or None")
    if not _integer(max_bytes) or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        entries = [name for name in names if name.endswith("/data.pkl")]
        if len(names) != len(set(names)) or len(entries) != 1:
            raise ValueError("one unambiguous state dictionary is required")
        root = entries[0][:-len("data.pkl")]
        if archive.getinfo(entries[0]).file_size > 8*1024**2:
            raise ValueError("pickle metadata exceeds size limit")
        if archive.getinfo(root + "byteorder").file_size > 6:
            raise ValueError("invalid byteorder record")
        byteorder = archive.read(root + "byteorder")
        if byteorder not in (b"little", b"big"):
            raise ValueError("unsupported byteorder")
        payload = archive.read(entries[0])
        # Extension-registry opcodes can bypass find_class through a process cache.
        if any(op.name in ("EXT1", "EXT2", "EXT4") for op, _, _ in pickletools.genops(payload)):
            raise pickle.UnpicklingError("pickle extension registry is forbidden")
        state = _Restricted(payload, max_bytes).load()
        if not isinstance(state, (dict, OrderedDict)) or len(state) > 10000:
            raise ValueError("bounded plain state dictionary required")
        result, used = {}, 0
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, _Tensor):
                raise ValueError("state entries must be named plain tensors")
            storage = tensor.storage
            dtype = np.dtype(("<" if byteorder == b"little" else ">") + storage.dtype)
            member = root + "data/" + storage.key
            if archive.getinfo(member).file_size != storage.count*dtype.itemsize:
                raise ValueError("storage byte count disagrees with metadata")
            if prefix is not None and not name.startswith(prefix):
                continue
            size = dtype.itemsize
            for dimension in tensor.shape:
                size *= dimension
            used += size
            if used > max_bytes:
                raise ValueError("selected tensor copies exceed size limit")
            data = archive.read(member)
            array = np.ndarray(tensor.shape, dtype=dtype, buffer=data,
                               offset=tensor.offset*dtype.itemsize,
                               strides=tuple(s*dtype.itemsize for s in tensor.strides))
            result[name if prefix is None else name[len(prefix):]] = array.copy()
        return result
