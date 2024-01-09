# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide extensions to native Python container data types and those in the standard Python
'collections' package.
"""
import re

from itertools import chain
from collections.abc import Mapping
from collections import defaultdict
from typing import Iterable
from contextlib import suppress


class SimpleNamespace:
    """
    Simple plain ol' data class.

    .. note::
     * Differs from `types.SimpleNamespace` in that an instance of this class is considered empty
       if it contains no members (i.e., bool(types.SimpleNamespace()) is always True, inconveniently)
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


def dictify(obj, magic=False, private=True, pod=type(None), elemfilter=None):
    """
    Converts an object, and any/all nested subobjects it contains, to a deep dictionary representation of that object.

    :param obj:        Object to convert
    :param magic:      "Include "magic" (dunder) elements of (sub-)object(s) in result."
    :type  magic:      bool
    :param private:    "Include "private" (name-prefixed with underscore) elements of (sub-)object(s) in result."
    :type  private:    bool
    :param pod:        How to ensure result elements are all "plain ol' data" (Ellipsis => allow "non-data" elements) --
                       tuple:
                         [0]: Sentinel value substituted for non-data list/tuple elements or simple values in result
                              (non-data items are omitted from dictionaries)
                         [1]: (optional) Single-argument callable to map a non-data element or simple value to a
                              substitute value in result (unspecified => "non-data" defined as any callable value)
    :type  pod:        Union(Iterable, Ellipsis)
    :param elemfilter: Single-argument callable taking an element value and returning the boolean "include this value." 
    :type  elemfilter: Union(Callable, None) 

    :return: Deeply nested "dictionary representation" of `obj`: a native Python aggregation (dict/list/tuple) if `obj`
             is not simple) or the (optionally POD-substituted) `obj` value itself if it is simple

    .. note::
     * WARNING: This function uses recursion internally to convert `obj` and all its elements (if any) to a
       deep dictionary representation; basic cycle detection is employed to prevent bottomless recursion.
     * Each (sub-)object is converted to a dict iff it contains a `__dict__` attribute (provided it not a callable);
       otherwise, it remains invariant.
    """
    def _dictify(_obj, _objmap):
        """ Dictifier: called recursively element-by-element to convert to suitable representation. """
        if not hasattr(_obj, '__call__') and any(b.__name__ in __builtins__ for b in _obj.__class__.__bases__):
            _obj = getattr(_obj, '__dict__', _obj)
        if isinstance(_obj, dict):
            _objmap.add(id(_obj))
        if isinstance(_obj, dict):
            _result = {_key: _dictify(_val, _objmap) for _key, _val in _obj.items()
                       if (elemfilter(_val) and isinstance(_key, str) and
                           (not _key.startswith('_') or (magic if _key.startswith('__') else private)) and
                           id(_val) not in _objmap and _pod_subst(_val) is not default)}
        elif isinstance(_obj, Iterable) and not isinstance(_obj, (str, bytes)):
            _result = (tuple, list)[isinstance(_obj, list)](_dictify(_pod_subst(_val), _objmap) for _val in _obj
                                                            if elemfilter(_val))
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
                      lambda _elem: _elem if type(_elem).__name__ in __builtins__ else default)
        if not callable(_pod_subst):
            raise TypeError("POD substitutor function is not callable")

    if not elemfilter:
        elemfilter = lambda *_: True

    return _dictify(obj, set())


def objectify(_dict):
    """
    Converts a nested dictionary into an object tree at every nesting level.

    .. note::
     * WARNING: Uses recursion to descend nesting levels.
    """
    return OmniDict(**{key: objectify(val) if isinstance(val, dict) else val for key, val in _dict.items()})


def objassign(obj, _dict):
    """
    Assigns items from a dictionary as attributes to an object tree, if assignable.
    """
    outobj = OmniDict(**vars(obj))
    outdict = {}
    for key, value in _dict.items():
        try:
            setattr(outobj, key, value)
        except (Exception, BaseException):
            outdict[key] = value

    for key in list(vars(outobj).keys()):
        if '.' in key:
            outdict[key] = getattr(outobj, key)
            delattr(outobj, key)

    return outobj, outdict


def deepupdate(dict1, dict2):
    """
    Merges nested dictionaries at every nesting level: items from the second dictionary supplement/override those
    from the first at the corresponding level.

    .. note::
     * The first dictionary is assigned the merged dictionary as a side-effect.
    """
    for key, val in dict2.items():
        dict1[key] = deepupdate(dict1[key], dict2.get(key, {})) if isinstance(val, dict) and key in dict1 else val
    return dict1


def deepkeys(obj, prefix='', sep='.'):
    """
    Extracts all fully-qualified (delimited) item keys from a nested object.
    """
    keys = []
    for key, val in vars(obj).items():
        fullkey = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(val, dict) or hasattr(val, '__dict__'):
            keys.extend(deepkeys(val, prefix=fullkey))
        else:
            keys.append(fullkey)
    return keys


def deepget(obj, attrspec, sep='.', default=NotImplemented):
    """
    Retrieves an attribute or item value from a nested object or dictionary as specified by a delimited
    attribute/item specification.
    """
    attrval = obj if isinstance(obj, dict) else vars(obj) if obj else {}
    item = attrval.get(attrspec, default)
    if item != default:
        attrval = item
    else:
        subattr, *rem = attrspec.split(sep, maxsplit=1)
        attrval = attrval.get(subattr, default)
        if attrval != default and rem:
            attrval = deepget(attrval, rem[0], default=default)
    return attrval


def deepassign(obj, attrspec, value, sep='.', default=NotImplemented):
    """
    Assigns an attribute or item value within a nested object or dictionary as specified by a delimited
    attribute/item specification.
    """
    attrval = obj if isinstance(obj, dict) else vars(obj) if obj else {}
    item = attrval.get(attrspec, default)
    if item == default:
        subattr, *rem = attrspec.split(sep, maxsplit=1)
        item = attrval.get(subattr, default)
        if item != default and rem:
            attrval, attrspec = item, rem[0]
            item = deepget(attrval, attrspec, default=default)
    if item != default:
        attrval[attrspec] = value

    return attrval


def deepsubst(_dict, _top=None):
    """
    Performs symbolic substitutions for self-references at every level of a nested object/dictionary.
    """
    if _top is None:
        _top = _dict
    patt = re.compile(r"(.*)\${([\w.]+)}(.*)")
    maxdepth = 10
    for _key, _val in _dict.items():
        if isinstance(_val, str):
            for _ in range(maxdepth):
                match = re.match(patt, _val)
                if not match:
                    break
                pref, sub, suff = match.groups()
                try:
                    # noinspection PyTypeChecker
                    sub = deepget(_top, sub, default='')
                    if isinstance(sub, str):
                        _val = _dict[_key] = ''.join((pref, sub, suff))
                    else:
                        _val = sub
                        break
                except (Exception, BaseException) as exc:
                    raise ValueError(f"Symbolic substitution error for ('{_key}', '{_val}')): {exc}") from exc
            else:
                raise ValueError(f"Configuration symbol substitution nesting depth exceeded, partial result: {_val}")
        if isinstance(_val, dict):
            _dict[_key] = deepsubst(_val, _top=_top)
        elif isinstance(_val, list):
            for _i, _item in enumerate(_val):
                _val[_i] = deepsubst({None: _item}, _top=_top)[None]
            _dict[_key] = _val
    return _dict


def nest(obj, sep='.'):
    """
    Creates a nested object tree or dictionsry keyed by simple item names from a simple non-nested dictionary whose
    keys are hierarchically delimited names.

    .. note::
     * WARNING: Uses recursion to construct nesting levels hierarchically.
     * An attempt is made to match the data type of each leaf object/dictionary in the resulting tree to be the same
       type as that of the flat input object/dictionary.
    """
    def _deepcast(_inptype, _tree):
        """ Casts a result tree to match the type of the specified input, at every nesting level. """
        try:
            _tree = _inptype(_tree)
        except (Exception, BaseException):
            with suppress(Exception):
                _tree = _inptype(**_tree)
        for _k, _v in _tree.items():
            if isinstance(_v, defaultdict):
                _tree[_k] = _deepcast(_inptype, _v)
        return _tree

    def _nest_node(_input, _result=None):
        """ Constructs a nested node in the result tree. """
        if _result is None:
            _result = defaultdict(dict)
        _dict = getattr(_input, '__dict__', _input)
        for _key, _val in _dict.items():
            _path, *_subpath = _key.split(sep, maxsplit=1)
            if _subpath:
                _nest_node({_subpath[0]: _val}, _result=_result.setdefault(_path, defaultdict(dict)))
            else:
                _result[_path] = _val
        return _result

    return _deepcast(type(obj), _nest_node(obj))


def flatten(obj, sep='.', prefix=''):
    """
    Flattens a possibly nested object tree or dictionary into a simple non-nested dictionary.

    .. note::
     * For dictionaries keyed by identifiers, the resulting keys in the flattened directory are composed hierarchically
       by the specified delimiter `sep`, unless `sep` is specified as None.
     * Key conflicts that might occur within the hierarchy are resolved by later items (in traversal order) superseding
       earlier ones in the resulting flattened dictionary.
    """
    result = {}
    _dict = getattr(obj, '__dict__', obj)
    for key, val in _dict.items():
        fullkey = f"{prefix}{sep}{key}" if sep is not None and prefix else key
        if isinstance(val, dict) or hasattr(val, '__dict__'):
            result.update(flatten(val, sep=sep, prefix=fullkey))
        else:
            result[fullkey] = val
    return result


def flatten_iter(iterable):
    """
    Flattens a possibly nested iterable (list- or tuple-ilke object) into a simple non-nested iterable.

    .. note::
     * An attempt is made to transfer the input type to the result type, defaulting to a list in the case of failure.
     * Duplicate elements in the resulting flattened iterable are not removed.
    """
    result = []
    for val in iterable:
        if isinstance(val, (str, bytes)) or not isinstance(val, Iterable):
            result.append(val)
        else:
            result.extend(flatten_iter(val))
    with suppress(Exception):
        result = type(iterable)(result)
    return result


if __name__ == '__main__':
    pass
    # # Unit test-ish:
    # xyz = OmniDict(a=1, b='xyz', c=None, dl=(OmniDict(xx='F', xy='M'),))
    # xyz.update(OmniDict(k1='K1', a=2), k2='k2', k3='k3', k1='k1')
    # sn = SimpleNamespace(abc=123, xyz=xyz, ssn=SimpleNamespace(), func=[dictify], nest=dict(func=lambda: None),
    #                      liter=iter([1, 2, 5]), cls=OmniDict, _priv=999, __xyz__=666)
    # dsn = dictify(sn)
    # print(dsn)
    # sn.liter = iter([1, 2, 5])  # pylint:disable=attribute-defined-outside-init
    # dsn = dictify(sn, pod=(None, lambda v: None if callable(v) else v), private=False, magic=True)
    # print(dsn)
    #
    # nd = OmniDict(a=1, b='xyz', c=None, dl=OmniDict(xx='F', xy='M'), d2=dict(abc='xyz', pdq=123))
    # fnd = flatten(nd)
    # print(fnd)
