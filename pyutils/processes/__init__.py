# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2018-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

""" Multiprocessing (native processes) helper functions. """

# noinspection PyPackageRequirements
import psutil


# pylint:disable=invalid-name
def kill_proc(pid, timeout, allow_suicide=False, kill_children=True):
    """ Takes extra measures to ensure a child process is killed (optionally along with all its children). """
    ok = pid == psutil.Process().pid
    if not ok or allow_suicide:  # (prevent suicides, unless from Kevorkian)
        ok = True
        try:
            proc = psutil.Process(pid)
        except (Exception, BaseException):
            proc = None
        if proc:
            if kill_children:
                for child in proc.children(recursive=True):
                    child.kill()
            for _try in range(2):
                _ = proc.terminate() if _try == 0 else proc.kill()
                if proc.wait(timeout=timeout) is not None:
                    break
            else:
                ok = False
    return ok
