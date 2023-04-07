# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide commonly-used iteration algorithms beyond those in the
standard Python itertools package.

.. note::
 * Use doctest to run executable examples within docstrings.
"""
# noinspection PyUnresolvedReferences
from collections import defaultdict
from collections.abc import (Generator, Iterator)

import itertools
try:
    # noinspection PyUnresolvedReferences
    from itertools import izip_longest as zipl
except ImportError:
    # noinspection PyUnresolvedReferences
    from itertools import zip_longest as zipl
import inspect


def list_append(alist, item):
    """ Python native list append(), with returned list. """
    alist.append(item)
    return alist


def list_remove(alist, item, every=False):
    """ Python native list remove(), with returned list. """
    if item in alist:
        if every:
            alist = [elem for elem in alist if elem != item]
        else:
            alist.remove(item)
    return alist


def list_extend(alist, other):
    """ Python native list extend(), with returned list. """
    alist.extend(other)
    return alist


def isiterable(entity):
    """
    Determines if an object is iterable.

    .. note::
     * It is insufficient to check if an object has some specific dunder
       attribute or combination thereof to determine if it is iterable.
    """
    try:
        iter(entity)
        result = True
    except TypeError:
        result = False
    return result


def issimple(entity):
    """
    Determines if an entity is "simple" -- i.e., a non-iterable or a string.

    .. note::
     * This does not traverse all elements of the specified entity, so will
       work with iterators and generators.
    """
    return isinstance(entity, str) or not isiterable(entity)


def isflat(entity):
    """
    Determines whether an entity is "flat" -- i.e., is either itself "simple"
    or an iterable composed completely of elements that are "simple".

    .. note::
     * This will traverse through all elements of the specified entity if it
       is an iterable, so is incompatible with iterators and generators.
    """
    # noinspection PyCallingNonCallable
    return (True if issimple(entity) else
             None if isinstance(entity, (Generator, Iterator)) else  # noqa: E127
             all(issimple(item) for item in (entity.values()  # noqa: E127
                 if hasattr(entity, 'values') else entity)))


def getlen(entity, default=None):
    """ Retrieves length of a Python iterable if it has length, else None. """
    return len(entity) if hasattr(entity, '__len__') else default


def index(haystack, needle, *args, **kwargs):
    """
    Safe (non-exception-generating) iterable indexer.

    :param haystack: Iterable to index within
    :param needle:   Element to index within `haystack`

    Optional args:
     * param start    Index at which to start the search (default: 0)
     * param end      Index at which to end the search (default: -1)
    Optional kwargs:
     * param default: Default value returned if `needle` is not found
     * param last:    "Find final index of `needle`."  (else find first)

    :return: Index of element found from iterable (or `default` if not found)

    .. example::
    >>> index([1, 2, 5, 2, 1], 2)
    1
    >>> index([1, 2, 5, 2, 1], 2, last=True)
    3
    >>> index([1, 2, 5, 2, 1], 3)
    None
    >>> index([1, 2, 5, 2, 1], 3, default=-1)
    -1
    >>> index("on waldon pond", 'waldo', 4, default=-1)
    -1
    >>> index("on waldon pond", 'on', 1, 9)
    7

    .. note::
     * This is notably absent from the native Python iterator typeology.
     * If `last` is specified, iterable must potentially be traversable multiple
       times (e.g., unsuitable for iterators and generators).
    """
    idx = kwargs.get('default', None)
    if not kwargs.get('last', False) and hasattr(haystack, 'index'):
        try:
            # noinspection PyCallingNonCallable
            idx = haystack.index(needle, *args)
        except (IndexError, ValueError):
            pass
    else:
        args = list(args or [0])
        while True:
            try:
                idx = haystack.index(needle, *args)
                args[0] = idx + 1
            except (IndexError, ValueError):
                break
    return idx


def getitem(haystack, needle, default=None):
    """
    Indexes or dereferences an element or attribute from an object, collection,
    or dictionary, returning a (specifiable) default value of the lookup fails.

    :param haystack: Indexable sequence/dictionary/object to index/dereference in
    :type  haystack: Union(Sequence, dict, object)
    :param needle:   Index or attribute to look up
    :type  needle:   Hashable
    :param default:  Default value if lookup fails (unspecified => None)
    """
    try:
        elem = haystack.__getitem__(needle)
    except (Exception, BaseException):
        try:
            elem = getattr(haystack, needle, default)
        except (Exception, BaseException):
            elem = default
    return elem


def find(selector, iterable, default=None, value=True):
    # noinspection PyTypeChecker
    """
    Evaluates a predicate selector function repeatedly over all elements
    of an iterable until the selector's match criteria is met.

    :param selector: "truthy" Iterable, or function accepting a single parameter
                     that is passed each successive element from `iterable`
                     and returns "truthy" value, indicating "successful match".
    :param iterable: Iterable to traverse
    :param default:  Default value returned if no match is found
    :param value:    True  => return the matched element
                     False => return the index of the match

    :return: Element found from iterable (or `default` if not found)

    .. example::
    >>> find(lambda x: (x % 3) == 0, [1, 2, 9, 3, 2])
    9
    >>> find(lambda x: (x % 3) == 0, [1, 2, 9, 3, 2], value=False)
    2
    >>> find(lambda x: (x % 5) == 0, [1, 2, 9, 3, 2])
    None
    >>> find(lambda x: (x % 5) == 0, [1, 2, 9, 3, 2], default=-1)
    -1
    >>> find((0, 1, 1, 0, 0, 0, 0, 1, 0), [1, 2, 9, 3, 2])
    2

    .. note::
     * This is analogous to the native Python filter() function, but returns
       the first match found rather than all matches.
    """
    result = default
    for idx, elem in enumerate(iterable):
        if selector(elem) if callable(selector) else selector[idx]:
            result = elem if value else idx
            break
    return result


def join(iterable, delim=None):
    """
    Joins a sequence of iterables into a flat list of all elements from those contained
    iterables; analogous to string.join().

    :param iterable: Iterable containing a sequence of iterables
    :param delim:    Iterable or element specifying the delimiter element to interleave
                     between groups of outermost elements

    :return: Joined list of elements from iterables
    :rtype:  list

    .. example::
    >>> join([[1,0,0,1,0], (0,0,1), [1,0]])
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    >>> join([[1,0,0,1,0], [0,0,1], [1,0]], delim=[None])
    [1, 0, 0, 1, 0, None, 0, 0, 1, None, 1, 0]
    """
    def _join(_iterable):
        return list(itertools.chain.from_iterable(_iterable))

    if delim is not None and not isiterable(delim):
        delim = (delim,)
    return (_join(iterable) if delim is None else
            _join(_join(zipl(iterable, [], fillvalue=delim))[:-1]))


def group(keys, values):
    """
    Groups values together into a dictionary of lists, each sharing the same key.

    :param keys:   Iterable containing a sequence of keys denoting grouping
    :param values: Iterable of containing corresponding values for `keys`

    :return: dict of lists: each dictionary key is a unique value from `keys` and
                            each dictionary value is a list of values fron `values`

    .. example::
    # (`keys` match `values`)
    >>> group([1,5,0,1,0,0], ('aa', 'bb', 'cc', 'dd', 'ee', 'ff'))
    {0: ['cc', 'ee', 'ff'], 1: ['aa', 'dd'], 5: ['bb']}
    # (`keys` longer than `values`)
    >>> group([1,5,0,1,0,0,2], ('aa', 'bb', 'cc', 'dd', 'ee', 'ff'))
    {0: ['cc', 'ee', 'ff'], 1: ['aa', 'dd'], 2: None, 5: ['bb']}
    # (`keys` shorter than `values`)
    >>> group([1,5,0,1], ('aa', 'bb', 'cc', 'dd', 'ee', 'ff'))
    {0: ['cc'], 1: ['aa', 'dd'], 5: ['bb'], None: ['ee', 'ff']}
    """
    ldict = defaultdict(list)
    for i, (key, value) in enumerate(zipl(keys, values)):
        if i < len(values):
            ldict[key].append(value)
        else:
            # noinspection PyTypeChecker
            ldict[key] = None
    return ldict


def distrimap(funcs, arguments):
    """
    Produces a distributed mapping iterator: when traversed, iterator will
    invoke each successive function, passing each corresponding successive
    grouping of values as parameters.

    :param funcs:     Iterable containing sequence of functions (length-extended
                      and cycled as necessary); assume a one-tuple if a callable
    :param arguments: Iterable containing sequence of argument iterables

    :return: iterable that will perform evaluations

    .. example::
    >>> list(distrimap(abs, [-1, +1, -2, +2]))
    [1, 1, 2, 2]
    >>> list(distrimap(int.__sub__, ((5, 4), (2, 6))))
    [1, -4]
    >>> list(distrimap((int.__add__, int.__sub__), ((5, 4), (2, 6))))
    [9, -4]
    >>> list(distrimap((int.__add__, int.__sub__), ((5, 4), (2, 6), (1, 2))))
    [9, -4, 3]
    >>> import random
    >>> list(distrimap(random.random, [[]]*3))
    [0.2573936231211953, 0.3296127325469308, 0.8879641606056986]
    >>> from functools import partial
    >>> list(distrimap(partial(random.choice, 'abcd'), [[]]*7))
    ['d', 'b', 'a', 'b', 'a', 'd', 'c']
    """
    return map((lambda _p: _p[0](*(_p[1] if isiterable(_p[1]) else _p[1:]))),
               zip(itertools.cycle(funcs if isiterable(funcs) else (funcs,)),
                   arguments))


def iterslice(aniter, *slicespec):
    """
    Indexes elements numerically from an iterable not normally indexable.

    :param aniter:    Iterable to extract numerical-indexed items from
    :param slicespec: Specification for slice indicating elements to index
                      (a slice or a tuple treated as a slice)
    :type  slicespec: Union(tuple,slice)

    :return: Numerically-indexed elements from specified iterable
             (a list if `slicespec` indexes multiple items)

    .. example::
    >>> ad = dict(a=123, b=456, c=789, d='abc').values()  # (assume ordered)
    >>> next(iter(ad))  # (the first item, natively)
    123
    >>> iterslice(ad, 0)  # (a single item, the first)
    123
    >>> iterslice(ad, 1)  # (a single item, not first)
    456
    >>> iterslice(ad, 1, 3)  # (range of items)
    [456, 789]
    >>> iterslice(ad, slice(1, 3))  # (same as above)
    [456, 789]
    >>> iterslice(ad, 0, None, 2)  # (ala slice)
    [123, 789]
    >>> iterslice(ad, -1, None)  # (negative indices ok if len() known)
    ['abc']
    """
    if isinstance(slicespec[0], slice):
        slicespec = (slicespec[0].start, slicespec[0].stop, slicespec[0].step)
    elif len(slicespec) == 1:
        slicespec = [slicespec[0], slicespec[0] + 1]
    def _zeronone(i):  # noqa: E306
        return 0 if i is None else i
    if any((_zeronone(s) < 0 for s in slicespec)):  # (handle negative indices)
        leniter = getlen(aniter)
        if leniter is None:
            raise IndexError('iterator has no known length')
        slicespec = type(slicespec)((s + leniter if _zeronone(s) < 0 else
                                     s for s in slicespec))
    items = list(itertools.islice(aniter, *slicespec))
    return items if not isinstance(slicespec, list) else items[0]


# noinspection PyUnboundLocalVariable,PyProtectedMember
def take(num, iterable, pad=NotImplemented):
    # noinspection PyTypeChecker,PyUnresolvedReferences
    """
    Returns a sequence containing the specified number of values, padded
    on left or right with a fixed pad value; result matches input type.

    :param num:      abs(num) => number of values to take; num<0 => pad on left
    :type  num:      int
    :param iterable: Iterable containing values to take
                     (scalar => assume a one-tuple)
    :param pad:      Pad value (unspecified => use default value for type of
                     first element of `iterable` or None if `iterable` is empty)

    :return: list or tuple of values, or string
    :rtype:  Iterable

    .. example::
    >>> take(6, [1,2,5])
    [1, 2, 5, 0, 0, 0]
    >>> take(-6, (1, 2, 5), pad=-1)
    (-1, -1, -1, 1, 2, 5)
    >>> take(5, 'abc')
    'abc  '
    >>> take(-5, 'abc', pad=None)
    [None, None, 'a', 'b', 'c']
    >>> take(5, range(7))
    range(5)
    >>> take(-5, range(7))
    range(2, 7)
    >>> take(-7, range(5))
    [0, 0, 0, 1, 2, 3, 4]
    >>> take(-5, (_i*2 for _i in range(3)))
    [None, None, 0, 2, 4]

    .. note::
     * If multiple pad values in the result are aggregates, they are individual
       copies, not aliases of each other.
     * Strings are space-padded unless `pad` is specified.
    """
    itlen = getlen(iterable, -1)
    if not hasattr(iterable, '__iter__') and not isinstance(iterable, str):
        iterable = (iterable,)

    reslen = abs(num)
    if isinstance(range, type) and isinstance(iterable, range) and reslen <= itlen:
        result = range(itlen + num if num < 0 else 0, num if num > 0 else itlen)
    else:
        isstr = isinstance(iterable, str)
        isleftpad = num < 0
        pad = (pad if pad is not NotImplemented else
               ' ' if isstr else
               type(iterable[0])() if itlen > 0 and iterable[0] is not None else
               None)
        if reslen > 0 > itlen:
            result = [pad] * reslen
            for i, elem in enumerate(iterable):
                if i == reslen:
                    break
                result[i] = elem
            # pylint:disable=undefined-loop-variable
            # noinspection PyUnboundLocalVariable
            num += i + 1
            if num < 0:
                # pylint:disable=undefined-loop-variable
                result = [pad] * -num + result[:i + 1]
        else:
            # pylint:disable=superfluous-parens
            result = (iterable[i] if 0 <= abs(i) < itlen + isleftpad else pad
                      for i in (range(num, 0) if isleftpad else range(num)))
            # noinspection PyProtectedMember,PyCallingNonCallable
            result = (''.join(result) if isstr and (isinstance(pad, str) or
                                                    reslen <= itlen) else
                      iterable._make(result) if hasattr(iterable, '_make') else
                      tuple(result) if isinstance(iterable, tuple) else
                      list(result))
    return result


# noinspection GrazieInspection
def xlat(value, spec):
    """
    Translates a value according to a substitution specification.

    :param value: Old value to translate
    :param spec:  Substitution pairs/dict or function:
                    * pairs: [0]: old value, [1]: new value
                    * dict: keys: old values, values: new values
                    * callable: arg: old value, result: new value
    :type  spec:  Union(Iterable, Callable, dict)

    :return: Translated value

    .. example::
    >>> xlat('doo', (('yabba', 'dabba'), ('doo', 'doody')))
    doody
    >>> xlat('foo', [('yabba', 'dabba'), ('doo', 'doody'), (None, '-none-'))]
    -none-
    >>> xlat("One World Is Enough", lambda s: s.lower().replace(' ', '_'))
    one_world_is_enough
    >>> xlat((1,2,5), {(1,2,3): range(1,4), (1,2,5): 'found', 'abc': 'xyz'})
    found
    >>> xlat('xyz', {(1,2,3): range(1,4), None: lambda x: isinstance(x, tuple)})
    False

    .. note::
     * If `spec` is specified as pairs or a dict and it contains a pair[0]
       value or key of None, then when the old value is not found in the spec,
       the corresponding pair[1] or key value is either the value to be used
       or a callable to perform the translation as the result.

    """
    if callable(spec):
        value = spec(value)
    elif isinstance(spec, dict):
        newstr = spec.get(value, NotImplemented)
        if newstr is not NotImplemented:
            value = newstr
        else:
            default = spec.get(None, NotImplemented)
            if default is not NotImplemented:
                value = default(value) if callable(default) else default
    else:
        default = NotImplemented
        for pair in spec:
            if pair[0] == value:
                value = pair[1]
                break
            if pair[0] is None:
                default = pair[1]
        else:
            if default is not NotImplemented:
                value = default(value) if callable(default) else default
    return value


# noinspection PyUnboundLocalVariable
def partition(selector, iterable, after=True, cycle=False):  # noqa: C901
    """
    Partitions a sequence of values into a list of sublists at "breakpoints"
    specified by a selection mask or set of indices,

    :param selector: Mask or indices indicating elements within `iterable` where
                     to partition it into sublists
                     (`cycle` applies if it is a mask shorter than `iterable`)
    :type  selector: Iterable<Union(int,bool)>
    :param iterable: Iterable to partition into sublists
    :param after:    "Partition after mask/index element."  (else before)
    :param after:    bool
    :param cycle:    "Cycle through mask elements."  (else False-extend mask)
    :param cycle:    bool

    :return: List of sublists, representing a partitioning of `iterable`
    :rtype:  list

    .. example::
    >>> T, F = True, False
    >>> partition((T, F, F, F,  T,  F,  T,  T,  F,  T),
                  [0, 1, 3, 4, 10, 11, 20, 30, 31, 40])
    [[0], [1, 3, 4, 10], [11, 20], [30], [31, 40]]
    >>> partition([0, 4, 6, 7, 9], (0, 1, 3, 4, 10, 11, 20, 30, 31, 40))
    ([0], [1, 3, 4, 10], [11, 20], [30], [31, 40])
    >>> partition(lambda _i: _i % 10 == 0, [0, 1, 3, 4, 10, 11, 20, 30, 31, 40])
    [[0], [1, 3, 4, 10], [11, 20], [30], [31, 40]]
    >>> partition(lambda c: c in ':-', 'ab-cdef:efg-hijkl')
    ['ab-', 'cdef:', 'efg-', 'hijkl']
    >>> partition(lambda c: c in ':-', 'ab-cdef:efg-hijkl', after=False)
    ['ab', '-cdef', ':efg', '-hijkl']
    >>> partition((False, True), 'abcdefghi')
    ['ab', 'cdefghi']
    >>> partition((False, True), 'abcdefghi', cycle=True)
    ['ab', 'cd', 'ef', 'gh', 'i']
    """
    isind = hasattr(selector, '__getitem__') and not isinstance(selector[0], bool)
    if isind:
        selector = sorted(selector)

    selen = getlen(selector)
    iterlen = getlen(iterable)
    if isind and selen is not None and iterlen is not None:
        result = [iterable[s]
                  for s in (slice(selector[i - 1] + after if i > 0 else 0,
                                  selector[i] + after if i < selen else None)
                            for i in range(selen + 1)
                            if i == selen or selector[i] <= iterlen - after)]
    else:
        def _castlast():
            if isinstance(iterable, tuple):
                # noinspection PyTypeChecker
                result[-1] = tuple(result[-1])
            elif isinstance(iterable, str):
                # noinspection PyTypeChecker
                result[-1] = ''.join(result[-1])

        def _partition():
            _castlast()
            result.append([])

        iscallable = callable(selector)
        if iscallable:
            isgenerator = inspect.isgeneratorfunction(selector)
            if isgenerator:
                # noinspection PyCallingNonCallable
                selector = selector()
                iscallable = False
        else:
            selector = itertools.cycle(selector) if cycle and not isind else iter(selector)

        if isind:
            selidx = next(selector)

        result = [[]]
        for elemidx, elem in enumerate(iter(iterable)):
            try:
                if iscallable:
                    # noinspection PyCallingNonCallable
                    part = selector(elem)
                elif isind:
                    # noinspection PyUnboundLocalVariable
                    part = elemidx == selidx
                    if part:
                        selidx = next(selector)
                else:
                    part = next(selector)
            except StopIteration:
                selidx = 0
            # noinspection PyUnboundLocalVariable
            if part and not after:
                _partition()
            result[-1].append(elem)
            if part and after:
                _partition()
            part = False

        if after:
            _castlast()
        else:
            _partition()

    if len(result[-1]) == 0:
        result = result[:-1]
    return result


# noinspection PyUnresolvedReferences,PyTypeChecker
def feedback(func, initial, feediter, **kwargs):  # noqa: C901
    """
    Performs non-recursive (iterative) "descent", feeding back the previous
    intermediate result into each successive invocation of a function/generator,
    also passing each of a succession of input values to the function/generator.

    :param func:     Function or generator to invoke repeatedly
    :type  func:     Callable
    :param initial:  "Result" value to feed into function on initial invocation
    :param feediter: Collection of argument values, each passed in succession
                     to each `func` invocation
    :type  feediter: Union(Iterable, Iterator)
    :param kwargs:   Optional keyword parameters, all passed to `func` every
                     invocation

    :return: End result of final invocation

    .. note::
     * `func` may accept either one argument or two: if one, the intermediate
       result and next successive `feediter` value are passed as a pair.
     * If `feediter` is an iterator, then to halt, `func` must either raise
       a StopIteration exception or return an iterable with `StopIteration`
       as the final item.
     * If `func` is a generator function (not a generator), the initial
       `feediter` value is passed as the second generator func argument, and
       each successive `feediter` value is sent to the generator func using the
       generator.send() method.  The generator func must receive each successive
       argument value via yield assignment and then yield each successive result
       value; a second yield is required for this value exchange.
     * If `func` is never invoked (e.g., due to a vacuous `feediter`), `initial`
       is returned.

    .. example::
    >>> nested_dict = {'a': {'b': {'c': 99}}}
    >>> feedback(getitem, nested_dict, ('a', 'b', 'c'))
    99
    >>> feedback(getitem, nested_dict, 'ab')
    {'c': 99}
    >>> rotate_right = lambda s, _: s[-1:] + s[:-1]  # (each passed arg ignored)
    >>> from itertools import repeat
    >>> feedback(rotate_right, 'trips', repeat('whatevah', 3))
    'ipstr'
    >>> feedback(rotate_right, 'trips', range(6))
    'strip'
    >>> from itertools import count
    >>> feedback(lambda r, i, n=0: r+[sum(r[-2:]) if r else 1] if i<=n else
    >>>                            (r, StopIteration),
    >>>          [], count(1), n=11)
    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    >>> from itertools import count
    >>> def digit_decade(intval, digit):  # (a generator function)
    >>>     for idx in count():
    >>>         yield (intval if intval > 0 and intval % 10 != digit else
    >>>                (10**idx if intval else 0, StopIteration))
    >>>         intval //= 10
    >>> feedback(digit_decade, 123579, 5))
    100
    >>> feedback(digit_decade, 123579, 2))
    10000
    >>> feedback(digit_decade, 123579, 4))
    0
    """
    result = initial
    try:
        if not isiterable(feediter):
            feediter = itertools.repeat(feediter)
        is_generator = isinstance(func, Generator)
        for arg in feediter:
            if not isinstance(func, Generator):
                try:
                    result = func(result, arg, **kwargs)
                except TypeError:
                    result = func((result, arg), **kwargs)
                if isinstance(result, Generator):
                    func = result
            if isinstance(func, Generator):
                if not is_generator:
                    try:
                        func.send(arg)
                    except TypeError:
                        pass
                result = func.__next__()
            if result is None:
                break
            try:
                if getlen(result) and result[-1] == StopIteration:
                    result = result[0]
                    raise StopIteration
            except StopIteration:
                raise
            except (Exception, BaseException):
                pass
    except StopIteration:
        pass
    return result


# noinspection PyPep8Naming
class scope:  # pylint:disable=invalid-name
    """
    Null context manager usable as a "break-able scope".

    .. note::
     * Also usable as the base for a context manager that needs premature exit capability.

    .. example::
    >>> with scope():
    >>>     print("always")
    >>>     raise StopIteration  # (equivalent to <break> in a Python-syntactic loop scope)
    >>>     # noinspection PyUnreachableCode
    >>>     print("never")
    always
    """
    def __enter__(self):
        pass

    def __exit__(self, exctype, _value, _traceback):
        return exctype is StopIteration
