"""
.. currentmodule:: arraycontext
.. autoclass:: PytatoArrayContext
"""
__copyright__ = """
Copyright (C) 2020-1 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from arraycontext.fake_numpy import \
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace
from arraycontext.context import ArrayContext
from arraycontext.container.traversal import \
        rec_multimap_array_container, rec_map_array_container
import numpy as np
from typing import Any, Callable, Tuple, Union, Sequence
from pytools.tag import Tag
from numbers import Number
import loopy as lp


class _PytatoFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    def norm(self, ary, ord=None):
        # FIXME: handle isinstance(ary, DOFArray) case
        return super().norm(ary, ord)


class _PytatoFakeNumpyNamespace(BaseFakeNumpyNamespace):
    def _get_fake_numpy_linalg_namespace(self):
        return _PytatoFakeNumpyLinalgNamespace(self._array_context)

    @property
    def ns(self):
        return self._array_context.ns

    def __getattr__(self, name):

        pt_funcs = ["abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                    "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                    "sqrt", "exp"]
        if name in pt_funcs:
            import pytato as pt  # type: ignore
            from functools import partial
            return partial(rec_map_array_container, getattr(pt, name))

        return super().__getattr__(name)

    def reshape(self, a, newshape):
        import pytato as pt
        return rec_multimap_array_container(pt.reshape, a, newshape)

    def transpose(self, a, axes=None):
        import pytato as pt
        return rec_multimap_array_container(pt.transpose, a, axes)

    def concatenate(self, arrays, axis=0):
        import pytato as pt
        return rec_multimap_array_container(pt.concatenate, arrays, axis)

    def ones_like(self, ary):
        def _ones_like(subary):
            import pytato as pt
            return pt.ones(subary.shape, subary.dtype)

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.maximum, x, y)

    def minimum(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.minimum, x, y)

    def where(self, criterion, then, else_):
        import pytato as pt
        return rec_multimap_array_container(pt.where, criterion, then, else_)

    def sum(self, a, dtype=None):
        import pytato as pt
        if dtype not in [a.dtype, None]:
            raise NotImplementedError
        return pt.sum(a)

    def min(self, a):
        import pytato as pt
        return pt.amin(a)

    def max(self, a):
        import pytato as pt
        return pt.amax(a)

    def stack(self, arrays, axis=0):
        import pytato as pt
        return rec_multimap_array_container(pt.stack, arrays, axis)

    # {{{ relational operators

    def equal(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.equal, x, y)

    def not_equal(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.not_equal, x, y)

    def greater(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.greater, x, y)

    def greater_equal(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.greater_equal, x, y)

    def less(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.less, x, y)

    def less_equal(self, x, y):
        import pytato as pt
        return rec_multimap_array_container(pt.less_equal, x, y)

    def conj(self, x):
        import pytato as pt
        return rec_multimap_array_container(pt.conj, x)

    def arctan2(self, y, x):
        import pytato as pt
        return rec_multimap_array_container(pt.arctan2, y, x)

    # }}}


class PytatoCompiledOperator:
    def __init__(self, actx, pytato_program, input_id_to_name_in_program,
                 output_id_to_name_in_program, output_template):
        self.actx = actx
        self.pytato_program = pytato_program
        self.input_id_to_name_in_program = input_id_to_name_in_program
        self.output_id_to_name_in_program = output_id_to_name_in_program
        self.output_template = output_template

    def __call__(self, *args):
        import pytato as pt
        import pyopencl.array as cla
        from arraycontext import (is_array_container,
                                  rec_keyed_map_array_container)

        input_kwargs_to_loopy = {}

        # {{{ extract loopy arguments execute the program

        for pos, arg in enumerate(args):
            if isinstance(arg, np.number):
                input_kwargs_to_loopy[self.input_id_to_name_in_program[(pos,)]] = (
                    arg)
            elif is_array_container(arg):
                def _extract_lpy_kwargs(keys, ary):
                    if isinstance(ary, pt.array.DataWrapper):
                        processed_ary = ary.data
                    elif isinstance(ary, cla.Array):
                        processed_ary = ary
                    elif isinstance(ary, pt.Array):
                        processed_ary = (self.actx.freeze(ary)
                                         .with_queue(self.actx.queue))
                    else:
                        raise TypeError("Expect pt.Array or CL-array, got "
                                f"{type(ary)}")

                    input_kwargs_to_loopy[
                        self.input_id_to_name_in_program[(pos,)
                                                         + keys]] = processed_ary
                    return ary

                rec_keyed_map_array_container(_extract_lpy_kwargs, arg)
            else:
                raise NotImplementedError(type(arg))

        # {{{ the generated program might not have depended on some of the
        # inputs => do not pass those to the loopy kernel

        input_kwargs_to_loopy = {arg_name: arg
                                 for arg_name, arg in input_kwargs_to_loopy.items()
                                 if arg_name in (self.pytato_program
                                                 .program.default_entrypoint
                                                 .arg_dict)}

        # }}}

        evt, out_dict = self.pytato_program(queue=self.actx.queue,
                                            allocator=self.actx.allocator,
                                            **input_kwargs_to_loopy)

        evt.wait()

        # }}}

        def to_output_template(keys, _):
            return self.actx.thaw(out_dict[self.output_id_to_name_in_program[keys]])

        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


class PytatoArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`pytato` data types to represent
    the DOF arrays targeting OpenCL for offloading operations.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator

        A :mod:`pyopencl` memory allocator. Can also be None (default) or False
        to use the default allocator.
    """

    def __init__(self, queue, allocator=None):
        super().__init__()
        self.queue = queue
        self.allocator = allocator
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        return _PytatoFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def clone(self):
        return type(self)(self.queue, self.allocator)

    def empty(self, shape, dtype):
        raise ValueError("PytatoArrayContext does not support empty")

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(shape, dtype)

    def from_numpy(self, np_array: np.ndarray):
        import pytato as pt
        import pyopencl.array as cla
        cl_array = cla.to_device(self.queue, np_array)
        return pt.make_data_wrapper(cl_array)

    def to_numpy(self, array):
        cl_array = self.freeze(array)
        return cl_array.get(queue=self.queue)

    def call_loopy(self, program, **kwargs):
        from pytato.loopy import call_loopy  # type: ignore
        import pyopencl.array as cla
        entrypoint, = set(program.callables_table)

        # thaw frozen arrays
        kwargs = {kw: (self.thaw(arg) if isinstance(arg, cla.Array) else arg)
                  for kw, arg in kwargs.items()}

        return call_loopy(program, kwargs, entrypoint)

    def freeze(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if isinstance(array, pt.Placeholder):
            raise ValueError("freezing placeholder would return garbage valued"
                    " arrays")
        if isinstance(array, cla.Array):
            return array.with_queue(None)
        if not isinstance(array, pt.Array):
            raise TypeError("PytatoArrayContext.freeze invoked with non-pt "
                            f"array of type '{type(array)}'")

        prg = pt.generate_loopy(array, cl_device=self.queue.device)
        evt, (cl_array,) = prg(self.queue)
        evt.wait()

        return cl_array.with_queue(None)

    def thaw(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if not isinstance(array, cla.Array):
            raise TypeError("PytatoArrayContext.thaw expects CL arrays, got "
                    f"{type(array)}")

        return pt.make_data_wrapper(array.with_queue(self.queue))

    # }}}

    def compile(self, f: Callable[[Any], Any],
                inputs_like: Tuple[Union[Number, np.ndarray], ...]
                ) -> Callable[..., Any]:
        from arraycontext import (rec_keyed_map_array_container,
                                  is_array_container)
        import pytato as pt

        dict_of_named_arrays = {}
        output_naming_map = {}
        input_naming_map = {}

        def to_placeholder(input_like, pos):
            if isinstance(input_like, np.number):
                name = f"_actx_in_{pos}"
                input_naming_map[(pos, )] = name
                return pt.make_placeholder((), input_like.dtype, name)
            elif is_array_container(input_like):
                def _rec_to_placeholder(keys, ary):
                    name = f"_actx_in_{pos}_" + "_".join(str(key)
                                                       for key in keys)
                    input_naming_map[(pos,) + keys] = name
                    return pt.make_placeholder(ary.shape, ary.dtype,
                                               name)
                return rec_keyed_map_array_container(_rec_to_placeholder,
                                                     input_like)
            else:
                raise NotImplementedError("Unknown input type "
                                          f"'{type(input_like)}'.")

        outputs = f(*[to_placeholder(el, iel)
                      for iel, el in enumerate(inputs_like)])

        if not is_array_container(outputs):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise ValueError("Function to be compiled did not return an array"
                             " container.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + "_".join(str(key)
                                          for key in keys)
            output_naming_map[keys] = name
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                      outputs)

        pytato_program = pt.generate_loopy(dict_of_named_arrays,
                                           options={"return_dict": True},
                                           cl_device=self.queue.device)

        if False:
            # transforming leads to compile-time slow downs (turning off for now)
            pytato_program.program = self.transform_loopy_program(pytato_program
                                                                  .program)

        return PytatoCompiledOperator(self, pytato_program,
                                      input_naming_map,
                                      output_naming_map,
                                      output_template=outputs)

    def transform_loopy_program(self, prg):
        from loopy.translation_unit import for_each_kernel

        nwg = 48
        nwi = (16, 2)

        @for_each_kernel
        def gridify(knl):
            # {{{ Pattern matching inames

            for insn in knl.instructions:
                if isinstance(insn, lp.CallInstruction):
                    # must be a callable kernel, don't touch.
                    pass
                elif isinstance(insn, lp.Assignment):
                    bigger_loop = None
                    smaller_loop = None
                    for iname in insn.within_inames:
                        if iname.startswith("iel"):
                            assert bigger_loop is None
                            bigger_loop = iname
                        if iname.startswith("idof"):
                            assert smaller_loop is None
                            smaller_loop = iname

                    if bigger_loop or smaller_loop:
                        assert bigger_loop is not None and smaller_loop is not None
                    else:
                        sorted_inames = sorted(tuple(insn.within_inames),
                                key=knl.get_constant_iname_length)
                        smaller_loop = sorted_inames[0]
                        bigger_loop = sorted_inames[1]

                    knl = lp.chunk_iname(knl, bigger_loop, nwg,
                            outer_tag="g.0")
                    knl = lp.split_iname(knl, f"{bigger_loop}_inner",
                            nwi[0], inner_tag="l.1")
                    knl = lp.split_iname(knl, smaller_loop,
                            nwi[1], inner_tag="l.0")
                elif isinstance(insn, lp.BarrierInstruction):
                    pass
                else:
                    raise NotImplementedError

            # }}}

            return knl

        prg = lp.set_options(prg, "insert_additional_gbarriers")

        return gridify(prg)

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        if arg_names is not None:
            from warnings import warn
            warn("'arg_names' don't bear any significance in PytatoArrayContext.",
                 stacklevel=2)

        import pytato as pt
        import pyopencl.array as cla

        def preprocess_arg(arg):
            if isinstance(arg, cla.Array):
                return self.thaw(arg)
            else:
                assert isinstance(arg, pt.Array)
                return arg

        return pt.einsum(spec, *(preprocess_arg(arg) for arg in args))


# }}}