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


class SimpleNamespace:  # (types.SimpleNamespace is incompatible)
    """ Simple data class. """
    def __init__(self, *members, **supplmembs):
        """ Initialize from a dictionary as a single positional parameter and/or keyword parameters. """
        for k, v in chain((members[0] if members else {}).items(), supplmembs.items()):
            setattr(self, k, v)

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
                list(self.keys()) == list(other.keys()) and all(self[k] == other[k] for k in self.keys()))

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

    def update(self, other):
        """ Adds all members from specified collection into this class instance. """
        for key, val in other.items():
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
