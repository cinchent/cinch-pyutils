# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide extensions to native Python container data types and those in the standard Python
'collections' package.
"""

from itertools import chain
from collections.abc import Mapping
from typing import Iterable


class SimpleNamespace:
    """
    Simple data class.

    .. note::
     * Differs from `types.SimpleNamespace` in that an instance of this class is considered empty
       if it contains no members (i.e., bool(types.SimpleNamespace()) is always True)
    """
    def __init__(self, *members, **supplmembs):
        """ Initialize from a dictionary as a single positional parameter and/or keyword parameters. """
        for key, val in chain((members[0] if members else {}).items(), supplmembs.items()):
            setattr(self, key, val)

    def __repr__(self):
        """ String formatter: enumerate attributes/values. """
        return f"""{self.__class__.__name__}({', '.join(f"{k}={repr(v)}" for k, v in self.__dict__.items())})"""

    def __bool__(self):
        """ Consider namespace instance empty if it contains no data members. """
        return bool(self.__dict__)


class OmniDict(SimpleNamespace, dict, Mapping):
    """
    Omnibus class whose instance objects contain members accessible/assignable by attribute (dot notation),
    named index, or numeric index.

    .. note::
     * This *is* a dict, but augmented with object-like and list-like properties.
    """
    def __init__(self, *members, _origin=0, **supplmembs):
        """
        Initializer: Creates dictionary-like hybrid object.

        :param members:     (optional single argument) Dictionary containing initial object members
        :param _origin:     Numeric index origin (i.e., starting index) for item indexing
        :type  _origin:     int
        :param supplmembs:  Dictionary containing supplemental object members

        .. note::
         * Object members are presumed to be ordered by assignment order (thus requires Python 3.7+).
         * `members` (if specified) precede `supplmembs` in assignment ordering.
        """
        self._origin = _origin
        super().__init__(*members, **supplmembs)

    def __getattribute__(self, attr):
        """ Override: Excludes internal variable(s) from __dict__ retrieval. """
        return ({k: v for k, v in object.__getattribute__(self, attr).items() if k != '_origin'}
                if attr == '__dict__' else object.__getattribute__(self, attr))

    def __repr__(self):
        """ Representation formatter: exclude internal member(s). """
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        """ String formatter: exclude internal member(s). """
        return str(self.__dict__)

    def __len__(self):
        """ Returns number of defined object members. """
        return len(self.keys())

    def __contains__(self, item):
        """ Equates attribute existence with item existence. """
        try:
            is_contained = hasattr(self, self._index(item))
        except IndexError:
            is_contained = False
        return is_contained

    def __getitem__(self, item):
        """ Retrieves a member value via indexing. """
        val = self.get(item, Ellipsis)
        if val is Ellipsis:
            raise IndexError("Item not found: {}".format(item))
        return val

    def __setitem__(self, item, val):
        """ Alters a member value via indexing. """
        setattr(self, self._index(item), val)

    def __delitem__(self, item):
        """ Removes a member by indexing. """
        delattr(self, self._index(item))

    def __eq__(self, other):
        return (isinstance(other, dict) and len(self) == len(other) and
                list(self.keys()) == list(other.keys()) and all(v == other[k] for k, v in self.items()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        """ Implements iterator for item iteration (e.g., using double-star operator). """
        for elem in self.items():
            yield elem

    def _index(self, item):
        """ Internal method: performs numeric indexing within object; members are numbered in insertion order. """
        if isinstance(item, int):
            try:
                item = tuple(self.keys())[(item - self._origin) if item >= 0 else item]
            except IndexError as exc:
                raise IndexError("Item not found: {}".format(item)) from exc
        return item

    def get(self, item, default=None):
        """ Retrieves a member value via indexing; returns a specifiable default value if not present. """
        if isinstance(item, int):
            try:
                val = tuple(self.values())[(item - self._origin) if item >= 0 else item]
            except IndexError:
                val = default
        else:
            val = vars(self).get(item, default)
        return val

    def copy(self):
        """ Creates distinct copy of this class instance. """
        return self.__class__(vars(self))

    # ---- dict semantic implementations:
    def keys(self):
        """ Returns all member names. """
        return vars(self).keys()

    def values(self):
        """ Returns all member values. """
        return vars(self).values()

    def items(self):
        """ Returns all member (name, value) pairs. """
        return vars(self).items()

    def clear(self, *args):
        """ Deletes all members. """
        return object.__getattribute__(self, '__dict__').clear(*args)

    def pop(self, *args):
        """
        Retrieves a member value via indexing and deletes the member; if member is not defined,
        returns a default value (if specified) or raises KeyError (if not).
        """
        item, *default = args
        return object.__getattribute__(self, '__dict__').pop(*([self._index(item)] + default))

    def popitem(self):
        """
        Retrieves the final member (name, value) and deletes the member; raises KeyError no members exist.
        """
        return object.__getattribute__(self, '__dict__').popitem()

    def update(self, other, **kwother):
        """ Adds all members from specified collection into this class instance. """
        for obj in chain((other, kwother)):
            for key, val in getattr(obj, '__dict__', obj).items():
                self[key] = val

    def setdefault(self, *args):
        """
        Retrieves a member value via indexing, first assigning the member value to a specifiable default
        if it is not already defined.
        """
        item, *default = args
        return object.__getattribute__(self, '__dict__').setdefault(*([self._index(item)] + default))

    @classmethod
    def fromkeys(cls, *args):
        """ Creates a new class instances from a collection of keys and a specifiable default. """
        keys, default, *_ = args + (None,)
        obj = cls()
        # noinspection PyTypeChecker
        for key in keys:  # pylint:disable=not-an-iterable
            obj[key] = default
        return obj


def dictify(obj, magic=False, private=True, pod=type(None)):
    """
    Converts an object, and any/all nested subobjects it contains, to a deep dictionary repreentation of that object.

    :param obj:     Object to convert
    :param magic:   "Include "magic" (dunder) elements of (sub-)object(s) in result."
    :type  magic:   bool
    :param private: "Include "private" (name-prefixed with underscore) elements of (sub-)object(s) in result."
    :type  private: bool
    :param pod:     How to ensure "plain ol' data" result elements (Ellipsis => allow "non-data" elements) -- tuple:
                     [0]: Sentinel value substituted for non-data list/tuple elements or simple values in result
                          (non-data items are omitted from dictionaries)
                     [1]: (optional) Single-argument callable to map a non-data element or simple value to a substitute
                          value in result (unspecified => "non-data" defined as any callable value)
    :type  pod:     Union(Iterable, Ellipsis)

    :return: Deeply nested "dictionary representation" of `obj`: a native Python aggregation (dict/list/tuple) if `obj`
             is not simple) or the (optionally POD-substituted) `obj` value itself if it is simple

    .. note::
     * WARNING: This function uses recursion internally to convert `obj` and all its elements (if any) to a
       deep dictionaty representation.
     * Each (sub-)object is converted to a dict iff it contains a __dict__ attribute; otherwise, it is invariant.
    """
    def _dictify(_obj):
        """ Dictifier: called recursively element-by-element to convert to suitable representation. """
        if not hasattr(_obj, '__call__'):
            _obj = getattr(_obj, '__dict__', _obj)
        if isinstance(_obj, dict):
            _result = {_key: _dictify(_val) for _key, _val in _obj.items()
                       if ((not _key.startswith('_') or (magic if _key.startswith('__') else private)) and
                           _pod_subst(_val) is not default)}
        elif isinstance(_obj, Iterable) and not isinstance(_obj, (str, bytes)):
            _result = (tuple, list)[isinstance(_obj, list)](_dictify(_pod_subst(_val)) for _val in _obj)
        else:
            _result = _pod_subst(_obj)
        return _result

    default, _pod_subst = NotImplemented, lambda _elem: _elem
    if pod is not type(None):  # noqa:E721
        try:
            default, *_pod_subst = pod
        except (Exception, BaseException):
            default, _pod_subst = pod, None
        _pod_subst = (_pod_subst[0] if _pod_subst else
                      lambda _elem: _elem if hasattr(__builtins__, type(_elem).__name__) else default)
        if not callable(_pod_subst):
            raise TypeError("POD substitutor function is not callable")

    return _dictify(obj)


if __name__ == '__main__':
    xyz = OmniDict(a=1, b='xyz', c=None, dl=(OmniDict(xx='F', xy='M'),))
    xyz.update(OmniDict(k1='K1', a=2), k2='k2', k3='k3', k1='k1')
    sn = SimpleNamespace(abc=123, xyz=xyz, ssn=SimpleNamespace(), func=[dictify], nest=dict(func=lambda: None),
                         liter=iter([1, 2, 5]), cls=OmniDict, _priv=999, __xyz__=666)
    dsn = dictify(sn)
    print(dsn)
    sn.liter = iter([1, 2, 5])  # pylint:disable=attribute-defined-outside-init
    dsn = dictify(sn, pod=(None, lambda v: None if callable(v) else v), private=False, magic=True)
    print(dsn)
