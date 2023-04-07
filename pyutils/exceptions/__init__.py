# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

""" Exception utility helper functions. """

import sys
import traceback
import inspect
import re
from contextlib import (contextmanager, suppress)
from collections import namedtuple
ExceptionInfo = namedtuple('ExceptionInfo', 'type instance traceback'.split())
CommonExceptionMessageFields = frozenset('message strerror reason'.split())


# noinspection PyBroadException
@contextmanager
def try_raise(*args, exception=None):
    """
    Within an exception handling context, tries a recovery action and re-raises
    the original exception if that action fails.  Syntactic sugar for:
        try:
            <statements>
        except:
            try:
                <recovery action>
            except:
                raise

    :param args: Exceptions for which to attempt recovery action
                 (empty => all exceptions)
    :param exception: Original exception to indicate as the direct cause
                      (None => re-raise as another exception occurrance)
    :type  exception: BaseException

    .. example::
    >>> try:
    >>>     was_bad = 1/0
    >>> except:
    >>>     with try_raise():
    >>>         was_bad = float('nan')
    >>> was_bad
    nan
    >>> class DivideByZero(ZeroDivisionError): pass
    >>> try:
    >>>     this_will_fail = 1/0
    >>> except (Exception, BaseException) as _exc:
    >>>     with try_raise(exception=DivideByZero(_exc)):
    >>>         this_will_fail_again = str(1/0)
    DivideByZero: division by zero
    The above exception was the direct cause of the following exception:
    ...
    ZeroDivisionError: division by zero
    """
    try:
        yield
    except (args if args else BaseException) as exc:
        if exception:
            raise exc from exception
        raise


def exception_message(exception):
    """
    Extractor for likeliest message text from an exception class object.

    :param exception: Exception class object
    :type  exception: UnionException
    """
    # noinspection PyUnusedLocal
    message = ''
    with suppress(Exception):
        message = str(exception)

    if not message:
        with suppress(Exception):
            for fld in CommonExceptionMessageFields:
                message = getattr(exception, fld, '')
                if message:
                    break
    if not message:
        with suppress(Exception):
            if isinstance(exception[0], str):
                message = exception[0]
    if not message:
        with suppress(Exception):
            message = exception.__class__.__name__

    return message


class ExceptionString(str):
    """ Thin cover for a string used to describe a thrown exception. """


# noinspection GrazieInspection
def throw(exc_type, logmethod=None, message='', exception=None):
    """
    Raises an exception, first optionally calling a logger method to log the
    specified message.

    :param exc_type:  Exception subclass/instance to raise
    :type  exc_type:  Union(Exception, type)
    :param logmethod: Level-specific logger method to call
                      (non-callable => skip logging)
    :param logmethod: Union(Callable, None)
    :param message:   Explanatory diagnostic message text
    :type  message:   str
    :param exception: Exception to propagate as "original cause" when
                      re-raising (Python 3 except "from" specification):
                        None => spontaneous exception
                        NotImplemented => not an error, don't log traceback
    :param exception: Union(BaseException, None, NotImplemented)

    .. note::
     * IMPORTANT: DOES NOT RETURN! This method always raises an exception.
     * The exception raised is an instance of auto-derived exception class that
       is composed from `exc_type` and the ThrownException mixin class,
       allowing nested tracebacks to be avoided when throw() appears multiple
       times in the same call stack.

    .. example::
    >>> from logging import getLogger
    >>> throw(TypeError, getLogger('').error, "dat ain' right")
    >>> class BadStringOperation(IndexError): pass
    >>> try:
    >>>     _ = 'abc'[4]
    >>> except IndexError as exc:
    >>>     throw(BadStringOperation, getLogger('').warning, "wrong!", exc)
    """
    try:
        already_thrown = isinstance(exc_type.args[0], ExceptionString)
    except (Exception, BaseException):
        already_thrown = False
    exc_class = (exc_type.__class__ if isinstance(exc_type, Exception) else
                 exc_type)

    if callable(logmethod):
        message = "{}: {}".format(exc_class.__name__, message)
        if exception is not NotImplemented and not already_thrown:
            message = format_traceback(message=message, stack=None)
        logmethod(message)

    if not already_thrown:
        message = ExceptionString(message)

    if isinstance(exception, BaseException):
        # noinspection PyCallingNonCallable
        raise exc_class(message) from exception
    # noinspection PyCallingNonCallable
    raise exc_class(message)


class TracebackString(str):
    """
    Thin wrapper for a formatted traceback string, as returned by
    `format_traceback()`, allowing it to be identified as such by a caller.
    """


def format_traceback(message='', stack=Exception):
    """
    Formats a traceback string for the occurrence of an exception, or for
    an arbitrary execution checkpoint.

    :param stack:   Traceback stack, ala `traceback.extract_stack()`:
                      Exception => use stack from most recent exception
                      None => current call stack (this frame pruned out)
    :type  stack:   list<FrameSummary>
    :param message: Explanatory message text to precede traceback stack
    :type  message: str

    :return: Multi-line traceback message
    :rtype:  str

    .. example::
    >>> traceback_str = format_traceback('checkpoint', stack=None)  # "here"
    >>> traceback_str.split('''\n''')[0]
    checkpoint
    >>> try:
    >>>     this_will_fail = 1/0
    >>> except (Exception, BaseException) as exc:
    >>>     traceback_str = format_traceback()
    >>> traceback_str.split('''\n''')[-2]
    ZeroDivisionError: division by zero
    """
    # noinspection PyUnusedLocal
    log_message = None
    with suppress(Exception):
        module_str = "/{}.".format(__name__.rsplit('.', maxsplit=1)[-1])
        newline = '\n'
        if stack is Exception:
            excinfo = sys.exc_info()
            traceback_stack = (traceback.format_exception(*excinfo)
                               if excinfo[0] is not None else [])
            excinfo = ["--- causing:" if t.startswith("\nThe above") else
                       t.strip() if not t.startswith(' ') else
                       "<{}>".format(t.split(newline)[0].strip())
                       for t in traceback_stack
                       if (not t.startswith('Traceback') and
                           module_str not in t)]
        else:
            if stack is None:
                stack = traceback.extract_stack()
            excinfo = [re.sub(r"<FrameSummary file (.*)(, line)(.*)>",
                              r'<File "\1"\2\3>', str(s))
                       for s in stack if module_str not in str(s)]
        log_message = ("{}\n----- [traceback:\n".format(message) +
                       ("{}\n".format(newline.join(excinfo)) if excinfo else '') +
                       "----- ]")
    return TracebackString(log_message)


def assert_trace(condition):
    """
    Functional substitution for 'assert' statement, prepending the code context
    of any assertion failure to the exception diagnostic.

    :param condition: Condition to assert (interpreted as a boolean)

    :raises AssertionError: when `condition` is False

    .. example::
    >>> assert_trace(True == False)
    AssertionError: caller_module.py:9999 True == False
    """
    if not condition:
        stack = inspect.getouterframes(inspect.currentframe())
        (_frame, filename, linenum, _funcname, lines, index) = stack[1]
        source = lines[index].strip()[len(stack[0][3]) + 1: -1] if lines else ''
        diagnostic = "{}:{} {}".format(filename, linenum, source)
        raise AssertionError(diagnostic)
