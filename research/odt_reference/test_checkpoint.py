"""In-memory adversarial file-boundary tests. No tensor library imports."""

import io
import pickle
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from research.odt_reference.checkpoint import load_checkpoint


def atom(value):
    return pickle.dumps(value, protocol=2)[2:-1]


def tensor_pickle(*, shape=(2,), strides=(1,), offset=0, count=2):
    storage = b"(" + atom("storage") + b"ctorch\nFloatStorage\n" + atom("0") + atom("cpu") + atom(count) + b"tQ"
    tensor = b"ctorch._utils\n_rebuild_tensor_v2\n(" + storage + atom(offset) + atom(shape) + atom(strides) + b"\x89NtR"
    return b"\x80\x02}" + atom("layer.weight") + tensor + b"s."


def archive(payload=None, data=None, byteorder=b"little"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("checkpoint/data.pkl", tensor_pickle() if payload is None else payload)
        zipped.writestr("checkpoint/byteorder", byteorder)
        zipped.writestr("checkpoint/data/0", np.array([3., 7.], dtype="<f4").tobytes() if data is None else data)
    stream.seek(0)
    return stream


class CheckpointBoundaryTests(unittest.TestCase):
    def test_load_raw_prefix_filter_and_copy(self):
        result = load_checkpoint(archive(), "layer.")
        np.testing.assert_array_equal(result["weight"], np.array([3., 7.], dtype=np.float32))
        self.assertTrue(result["weight"].flags.owndata)
        self.assertEqual(load_checkpoint(archive(), "absent."), {})

    def test_noncontiguous_bounded_view_and_byteorder(self):
        values = np.array([3., 5., 7.], dtype=">f4")
        result = load_checkpoint(archive(tensor_pickle(count=3, strides=(2,)), values.tobytes(), b"big"))
        np.testing.assert_array_equal(result["layer.weight"], [3., 7.])

    def test_arbitrary_global_and_extension_rejected_without_execution(self):
        with patch("os.system", side_effect=AssertionError("must not execute")) as forbidden:
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(archive(b"\x80\x02cos\nsystem\n" + atom("echo forbidden") + b"\x85R."))
            forbidden.assert_not_called()
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(archive(b"\x80\x02\x82\x01."))

    def test_storage_bounds_shape_stride_and_allocation_limits(self):
        bad = ({"offset": -1}, {"offset": 1}, {"strides": (-1,)}, {"strides": (2,)},
               {"shape": (3,)}, {"shape": (0,)}, {"shape": (2, 1)}, {"count": 0}, {"count": 100})
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_checkpoint(archive(tensor_pickle(**kwargs)), max_bytes=64)
        # Small backing storage cannot evade the aggregate copy limit via overlap.
        with self.assertRaises(ValueError):
            load_checkpoint(archive(tensor_pickle(shape=(20,), strides=(0,))), max_bytes=64)

    def test_storage_byte_count_and_unknown_byteorder(self):
        with self.assertRaises(ValueError):
            load_checkpoint(archive(data=b"short"))
        with self.assertRaises(ValueError):
            load_checkpoint(archive(byteorder=b"native"))


if __name__ == "__main__":
    unittest.main()
