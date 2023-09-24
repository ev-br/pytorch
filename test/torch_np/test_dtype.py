# Owner(s): ["module: dynamo"]

from unittest import expectedFailure as xfail

import numpy
import torch
import torch._dynamo
import torch._dynamo.testing

import torch._numpy as tnp

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


dtype_names = [
    "bool_",
    *[f"int{w}" for w in [8, 16, 32, 64]],
    "uint8",
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]

np_dtype_params = []

np_dtype_params = [
    subtest(("bool", "bool"), name="bool"),
    subtest(
        ("bool", numpy.dtype("bool")),
        name="numpy.dtype('bool')",
        decorators=[xfail],  # reason="XXX: np.dtype() objects not supported"),
    ),
]


for name in dtype_names:
    np_dtype_params.append(subtest((name, name), name=repr(name)))

    np_dtype_params.append(
        subtest((name, getattr(numpy, name)), name=f"numpy.{name}", decorators=[xfail])
    )  # numpy namespaced dtypes not supported
    np_dtype_params.append(
        subtest((name, numpy.dtype(name)), name=f"numpy.{name!r}", decorators=[xfail])
    )


@instantiate_parametrized_tests
class TestConvertDType(TestCase):
    @parametrize("name, np_dtype", np_dtype_params)
    def test_convert_np_dtypes(self, name, np_dtype):
        tnp_dtype = tnp.dtype(np_dtype)
        if name == "bool_":
            assert tnp_dtype == tnp.bool_
        elif tnp_dtype.name == "bool_":
            assert name.startswith("bool")
        else:
            assert tnp_dtype.name == name


@instantiate_parametrized_tests
class TestDTypesUnderDynamo(TestCase):

    @parametrize("dt", [subtest(float, name='py_float'),
                        subtest(numpy.float64, name='np.float64'),
                        subtest('float64', name="'float64'"),
                        subtest(numpy.dtype('float64'), name="dtype('float64')"),
                        # short floats
                        'float32',
                        subtest(numpy.float32, name='numpy.float32'),
                        subtest(numpy.dtype('float32'), name="dtype('float32'"),
                        # integers
                        subtest('int', name='str_int'),
                        subtest(int, name='py_int'),
                        subtest(numpy.intp, name='np.intp'),
                        subtest(numpy.int32, name='np.int32'),
                        subtest(numpy.uint8, name='np.uint8')])
    def test_with_dtype(self, dt):
        def fn(dt):
            return numpy.arange(5, dtype=dt)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        r = opt_fn(dt)
        assert cnts.frame_count == 1, f"frame_count = {cnts.frame_count}"



if __name__ == "__main__":
    run_tests()
