"""
.. _freeze-thaw:

Freezing and thawing
--------------------

One of the central concepts introduced by the array context formalism is
the notion of :meth:`~arraycontext.ArrayContext.freeze` and
:meth:`~arraycontext.ArrayContext.thaw`. Each array handled by the array context
is either "thawed" or "frozen". Unlike the real-world concept of freezing and
thawing, these operations leave the original array alone; instead, a semantically
separate array in the desired state is returned.

*   "Thawed" arrays are associated with an array context. They use that context
    to carry out operations (arithmetic, function calls).

*   "Frozen" arrays are static data. They are not associated with an array context,
    and no operations can be performed on them.

Freezing and thawing may be used to move arrays from one array context to another,
as long as both array contexts use identical in-memory data representation.
Otherwise, a common format must be agreed upon, for example using
:mod:`numpy` through :meth:`~arraycontext.ArrayContext.to_numpy` and
:meth:`~arraycontext.ArrayContext.from_numpy`.

.. _freeze-thaw-guidelines:

Usage guidelines
^^^^^^^^^^^^^^^^
Here are some rules of thumb to use when dealing with thawing and freezing:

-   Any array that is stored for a long time needs to be frozen.
    "Memoized" data (cf. :func:`pytools.memoize` and friends) is a good example
    of long-lived data that should be frozen.

-   Within a function, if the user did not supply an array context,
    then any data returned to the user should be frozen.

-   Note that array contexts need not necessarily be passed as a separate
    argument. Passing thawed data as an argument to a function suffices
    to supply an array context. The array context can be extracted from
    a thawed argument using, e.g., :func:`~arraycontext.get_container_context`
    or :func:`~arraycontext.get_container_context_recursively`.

What does this mean concretely?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Freezing and thawing are abstract names for concrete operations. It may be helpful
to understand what these operations mean in the concrete case of various
actual array contexts:

-   Each :class:`~arraycontext.PyOpenCLArrayContext` is associated with a
    :class:`pyopencl.CommandQueue`. In order to operate on array data,
    such a command queue is necessary; it is the main means of synchronization
    between the host program and the compute device. "Thawing" here
    means associating an array with a command queue, and "freezing" means
    ensuring that the array data is fully computed in memory and
    decoupling the array from the command queue. It is not valid to "mix"
    arrays associated with multiple queues within an operation: if it were allowed,
    a dependent operation might begin computing before an input is fully
    available. (Since bugs of this nature would be very difficult to
    find, :class:`pyopencl.array.Array` and
    :class:`~meshmode.dof_array.DOFArray` will not allow them.)

-   For the lazily-evaluating array context based on :mod:`pytato`,
    "thawing" corresponds to the creation of a symbolic "handle"
    (specifically, a :class:`pytato.array.DataWrapper`) representing
    the array that can then be used in computation, and "freezing"
    corresponds to triggering (code generation and) evaluation of
    an array expression that has been built up by the user
    (using, e.g. :func:`pytato.generate_loopy`).

The interface of an array context
---------------------------------

.. currentmodule:: arraycontext
.. autoclass:: ArrayContext
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

from typing import Sequence, Union, Callable, Any
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from pytools import memoize_method
from pytools.tag import Tag


# {{{ ArrayContext

class ArrayContext(ABC):
    r"""
    :canonical: arraycontext.ArrayContext

    An interface that allows software implementing a numerical algorithm
    (such as :class:`~meshmode.discretization.Discretization`) to create and interact
    with arrays without knowing their types.

    .. versionadded:: 2020.2

    .. automethod:: empty
    .. automethod:: zeros
    .. automethod:: empty_like
    .. automethod:: zeros_like
    .. automethod:: from_numpy
    .. automethod:: to_numpy
    .. automethod:: call_loopy
    .. automethod:: einsum
    .. attribute:: np

         Provides access to a namespace that serves as a work-alike to
         :mod:`numpy`.  The actual level of functionality provided is up to the
         individual array context implementation, however the functions and
         objects available under this namespace must not behave differently
         from :mod:`numpy`.

         As a baseline, special functions available through :mod:`loopy`
         (e.g. ``sin``, ``exp``) are accessible through this interface.

         Callables accessible through this namespace vectorize over object
         arrays, including :class:`arraycontext.ArrayContainer`\ s.

    .. automethod:: freeze
    .. automethod:: thaw
    .. automethod:: tag
    .. automethod:: tag_axis
    .. automethod:: compile
    """

    def __init__(self):
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import BaseFakeNumpyNamespace
        return BaseFakeNumpyNamespace(self)

    @abstractmethod
    def empty(self, shape, dtype):
        pass

    @abstractmethod
    def zeros(self, shape, dtype):
        pass

    def empty_like(self, ary):
        return self.empty(shape=ary.shape, dtype=ary.dtype)

    def zeros_like(self, ary):
        return self.zeros(shape=ary.shape, dtype=ary.dtype)

    @abstractmethod
    def from_numpy(self, array: np.ndarray):
        r"""
        :returns: the :class:`numpy.ndarray` *array* converted to the
            array context's array type. The returned array will be
            :meth:`thaw`\ ed.
        """
        pass

    @abstractmethod
    def to_numpy(self, array):
        r"""
        :returns: *array*, an array recognized by the context, converted
            to a :class:`numpy.ndarray`. *array* must be
            :meth:`thaw`\ ed.
        """
        pass

    def call_loopy(self, program, **kwargs):
        """Execute the :mod:`loopy` program *program* on the arguments
        *kwargs*.

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.TranslationUnit`.
        It is expected to not yet be transformed for execution speed.
        It must have :attr:`loopy.Options.return_dict` set.

        :return: a :class:`dict` of outputs from the program, each an
            array understood by the context.
        """

    @memoize_method
    def _get_scalar_func_loopy_program(self, c_name, nargs, naxes):
        from pymbolic import var

        var_names = ["i%d" % i for i in range(naxes)]
        size_names = ["n%d" % i for i in range(naxes)]
        subscript = tuple(var(vname) for vname in var_names)
        from islpy import make_zero_and_vars
        v = make_zero_and_vars(var_names, params=size_names)
        domain = v[0].domain()
        for vname, sname in zip(var_names, size_names):
            domain = domain & v[0].le_set(v[vname]) & v[vname].lt_set(v[sname])

        domain_bset, = domain.get_basic_sets()

        import loopy as lp
        from .loopy import make_loopy_program
        from arraycontext.transform_metadata import ElementwiseMapKernelTag
        return make_loopy_program(
                [domain_bset],
                [
                    lp.Assignment(
                        var("out")[subscript],
                        var(c_name)(*[
                            var("inp%d" % i)[subscript] for i in range(nargs)]))
                    ],
                name="actx_special_%s" % c_name,
                tags=(ElementwiseMapKernelTag(),))

    @abstractmethod
    def freeze(self, array):
        """Return a version of the context-defined array *array* that is
        'frozen', i.e. suitable for long-term storage and reuse. Frozen arrays
        do not support arithmetic. For example, in the context of
        :class:`~pyopencl.array.Array`, this might mean stripping the array
        of an associated command queue, whereas in a lazily-evaluated context,
        it might mean that the array is evaluated and stored.

        Freezing makes the array independent of this :class:`ArrayContext`;
        it is permitted to :meth:`thaw` it in a different one, as long as that
        context understands the array format.

        See also :func:`arraycontext.freeze`.
        """

    @abstractmethod
    def thaw(self, array):
        """Take a 'frozen' array and return a new array representing the data in
        *array* that is able to perform arithmetic and other operations, using
        the execution resources of this context. In the context of
        :class:`~pyopencl.array.Array`, this might mean that the array is
        equipped with a command queue, whereas in a lazily-evaluated context,
        it might mean that the returned array is a symbol bound to
        the data in *array*.

        The returned array may not be used with other contexts while thawed.

        See also :func:`arraycontext.thaw`.
        """

    @abstractmethod
    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* with the *tags* applied. *array*
        itself is not modified.

        .. versionadded:: 2021.2
        """

    @abstractmethod
    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* in which axis number *iaxis* has
        the *tags* applied. *array* itself is not modified.

        .. versionadded:: 2021.2
        """

    @memoize_method
    def _get_einsum_prg(self, spec, arg_names, tagged):
        import loopy as lp
        from .loopy import _DEFAULT_LOOPY_OPTIONS
        from loopy.version import MOST_RECENT_LANGUAGE_VERSION
        return lp.make_einsum(
            spec,
            arg_names,
            options=_DEFAULT_LOOPY_OPTIONS,
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            tags=tagged,
        )

    # This lives here rather than in .np because the interface does not
    # agree with numpy's all that well. Why can't it, you ask?
    # Well, optimizing generic einsum for OpenCL/GPU execution
    # is actually difficult, even in eager mode, and so without added
    # metadata describing what's happening, transform_loopy_program
    # has a very difficult (hopeless?) job to do.
    #
    # Unfortunately, the existing metadata support (cf. .tag()) cannot
    # help with eager mode execution [1], because, by definition, when the
    # result is passed to .tag(), it is already computed.
    # That's why einsum's interface here needs to be cluttered with
    # metadata, and that's why it can't live under .np.
    # [1] https://github.com/inducer/meshmode/issues/177
    def einsum(self, spec, *args, arg_names=None, tagged=()):
        """Computes the result of Einstein summation following the
        convention in :func:`numpy.einsum`.

        :arg spec: a string denoting the subscripts for
            summation as a comma-separated list of subscript labels.
            This follows the usual :func:`numpy.einsum` convention.
            Note that the explicit indicator `->` for the precise output
            form is required.
        :arg args: a sequence of array-like operands, whose order matches
            the subscript labels provided by *spec*.
        :arg arg_names: an optional iterable of string types denoting
            the names of the *args*. If *None*, default names will be
            generated.
        :arg tagged: an optional sequence of :class:`pytools.tag.Tag`
            objects specifying the tags to be applied to the operation.

        :return: the output of the einsum :mod:`loopy` program
        """
        if arg_names is None:
            arg_names = tuple("arg%d" % i for i in range(len(args)))

        prg = self._get_einsum_prg(spec, arg_names, tagged)
        return self.call_loopy(
            prg, **{arg_names[i]: arg for i, arg in enumerate(args)}
        )["out"]

    @abstractmethod
    def clone(self):
        """If possible, return a version of *self* that is semantically
        equivalent (i.e. implements all array operations in the same way)
        but is a separate object. May return *self* if that is not possible.

        .. note::

            The main objective of this semi-documented method is to help
            flag errors more clearly when array contexts are mixed that
            should not be. For example, at the time of this writing,
            :class:`meshmode.meshmode.Discretization` objects have a private
            array context that is only to be used for setup-related tasks.
            By using :meth:`clone` to make this a separate array context,
            and by checking that arithmetic does not mix array contexts,
            it becomes easier to detect and flag if unfrozen data attached to a
            "setup-only" array context "leaks" into the application.
        """

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """Compiles *f* for repeated use on this array context. *f* is expected
        to be a `pure function <https://en.wikipedia.org/wiki/Pure_function>`__
        performing an array computation.

        Control flow statements (``if``, ``while``) that might take different
        paths depending on the data lead to undefined behavior and are illegal.
        Any data-dependent control flow must be expressed via array functions,
        such as ``actx.np.where``.

        *f* may be called on placeholder data, to obtain a representation
        of the computation performed, or it may be called as part of the actual
        computation, on actual data. If *f* is called on placeholder data,
        it may be called only once (or a few times).

        :arg f: the function executing the computation.
        :return: a function with the same signature as *f*.
        """
        return f

    # undocumented for now
    @abstractproperty
    def permits_inplace_modification(self):
        pass

# }}}

# vim: foldmethod=marker
