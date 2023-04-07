# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to supplement native Python string processing.
"""
import sys
import json
import warnings
try:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from distutils.util import strtobool  # (deprecated)
    warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
except ImportError:
    def strtobool(val):  # (cut-n-paste from distutils.util)  # pylint:disable=too-many-return-statements
        """ Converts a string representation of truth to true (1) or false (0), or raises ValueError. """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):  # pylint:disable=no-else-return
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))


def str_to_dict(dictstr):
    # noinspection PyUnresolvedReferences
    """
    Converts a dictionary specification string into a dictionary constant.

    .. note::
     * This only converts text into a dictionary constant -- general expression evaluation is safeguarded against.

    .. example::
    >>> str_to_dict('''{"a": 123, "bc": "hello", "xyz": null}''')  # (strict JSON-compliant syntax)
    {'a': 123, 'bc': 'hello', 'xyz': None}
    >>> str_to_dict('''{'a': 123, 'bc': 'hello', "xyz": None}''')  # (Python dict constant syntax, agnostic quoting)
    {'a': 123, 'bc': 'hello', 'xyz': None}
    >>> str_to_dict("a=(1,2,3), bc='hello'")  # (Python syntax for arguments to 'dict()')
    {'a': (1, 2, 3), 'bc': 'hello'}
    >>> str_to_dict("a=123, bc=hello")  # (loose unbracketed comma-delimited key=value lists)
    {'a': '123', 'bc': 'hello'}
    >>> str_to_dict("a=123\nbc=hello")  # (or newline-delimited: for either, all values become strings, even numericals)
    {'a': '123', 'bc': 'hello'}
    """
    for _ in (True,):  # (break-able scope)
        try:
            _dict = json.loads(dictstr)  # (strict JSON)
            break
        except (Exception, BaseException):
            pass

        try:
            _dict = safe_eval(dictstr)  # (Python dict constant)
            break
        except (Exception, BaseException):
            pass

        try:
            _dict = safe_eval("dict({})".format(dictstr), symbols={'dict': dict})  # (Python dict() parameters)
            break
        except (Exception, BaseException):
            pass

        if '=' in dictstr:  # (comma-/newline-delimited key=str)
            try:
                dictstr = dictstr.replace('\n', ',')
                # noinspection PyTypeChecker
                _dict = dict((itemstr.strip()
                              for itemstr in item.split('=', maxsplit=1))
                             for item in dictstr.split(','))
                break
            except (Exception, BaseException):
                pass
    else:
        _dict = dictstr
    return _dict


def sysargv(argv=None, comment='#'):
    """
    Returns system command-line arguments up to an argument with a leading comment
    string.
    """
    if argv is None:
        argv = sys.argv
    comments = tuple(filter(lambda _p: _p[1].startswith(comment), enumerate(argv[1:])))
    return argv[:1 + comments[0][0]] if comments else argv


def safe_eval(expr, symbols=None):
    """
    Evaluates an expression safely, restricting symbols to only those specified.
    """
    # pylint:disable=eval-used
    return eval(expr, {'__builtins__': None}, symbols or {})


def truthy(value, truthstrs=None, strict=True):
    """
    Infers a boolean value from any value that is used to specify a predicate.

    :param value:     Value indicating a predicate ("numeric" scalar or string)
    :param truthstrs: Collection of (True, False) string pairs to recognize as
                      predicate (None => use distutils.util.strtobool())
    :type  truthstrs: Union(Iterable, None)
    :param strict:    "Raise exception if `value` unrecognized as truth string."
    :type  strict:    bool

    :return: "Specified value indicates a True predicate." (None => neither)
    :rtype:  Union(bool, None)
    """
    truth = bool(value) if isinstance(value, (bool, int, float)) else None
    if truth is None and isinstance(value, str):
        value = value.lower()
        if truthstrs:
            tstrs, fstrs = zip(*truthstrs)
            truth = (any(value == s.lower() for s in tstrs) or
                     (False if any(value == s.lower() for s in fstrs) else None))
        else:
            try:
                truth = bool(strtobool(value))
            except ValueError:
                truth = None
    if truth is None and strict:
        raise ValueError("invalid truth value %r" % (value,))
    return truth


def isascii(_str):
    """
    Determines if a Unicode string is wholly composed of ASCII characters.
    """
    try:
        _str.encode('ascii')
        asc = True
    except UnicodeEncodeError:
        asc = False
    return asc


def bytestr(_str):
    """ Python 3 conditional bytes-to-string decoder. """
    return str(_str.decode()) if isinstance(_str, bytes) else _str
