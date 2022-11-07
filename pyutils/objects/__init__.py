# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to supplement object handling capabilities present natively.

.. note::
* Use doctest to run executable examples within docstrings.
"""


def ident(obj):
    """ Vacuous function: returns self. """
    return obj


def default(value, default_value, sentinel=None):
    """
    Common idiom for substituting a default for a value that is a sentinel.

    :param value:         Value to test
    :param default_value: Default value to substitute if test is not satisfied
    :param sentinel:      Sentinel value (default: None) that is recognized and
                          substituted for

    :return: `value`, unless it matches `sentinel_value`, in which case
             `default_value`

    .. note::
     * To substitute a default <d> for a value <v> whenever <v> is not "truthy",
       use the well-known Python idiom 'v or d' instead (which also has the
       benefit being of lazily evaluated).
    """
    return value if value != sentinel else default_value


common_locals = ['self', '__class__', '_', 'kwargs']
def defmemb(this, argdict):  # noqa: E302
    # noinspection PyUnresolvedReferences
    """
    Common idiom to assign all items in a specified dictionary as members
    of the specified object.

    :param this:    Object to assign instance members of
    :param argdict: Dictionary specifying all member names and values to assign
                    (common locals (see above) if present, are not assigned)
    :type  argdict: dict

    .. note::
     * Common use would be for an object initializer to assign all its
       arguments as instance members of that object (e.g., from locals()).

    .. example::
    >>> class MyClass(object):
    >>>     def __init__(self, a, b, c, d='snarge'):
    >>>         defmemb(self, locals())
    >>> obj = MyClass(111, 222, 333)
    >>> obj.a
    111
    >>> obj.d
    'snarge'
    """
    for argname, argval in argdict.items():
        if argname not in common_locals:
            setattr(this, argname, argval)
