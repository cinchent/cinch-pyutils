# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2018-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to perform password authentication.
"""

import sys
import os
import subprocess
from contextlib import suppress
from base64 import b64decode


def password_plaintext(text):
    """ Extracts password plaintext from trivial (base-64) obfuscation (or assumes unobfuscated password). """
    plaintext = None
    if isinstance(text, bytes):
        text = text.decode()
    for inv in (None, -1):
        for sufflen in range(3):
            with suppress(Exception):
                plaintext = b64decode(text[::inv] + '=' * sufflen).decode()
                break
        if plaintext:
            break
    else:
        plaintext = text
    return plaintext


def password_validate_local(username=None, password=None):
    """
    Checks user password against local OS user account credentials.

    :param username: Username to authenticate against local OS user accounts (None => use USER envirosym)
    :type  username: Union(str, None)
    :param password: Password for user on local OS (None => use PASSWORD envirosym)
    :type  password: Union(str, None)

    :return: Validation failure reason text (None => success)
    :rtype:  Union(str, None)

    .. note::
     * IMPORTANT! This is an insecure way of retrieving and validating passwords; do not rely on this for
       mission-critical authentication or other protections.
     * Requires superuser privileges on POSIX platforms.
    """
    valid = "Incorrect username or password"
    if not username:
        username = os.getenv('USER', '')
    if not password:
        password = os.getenv('PASSWORD', '')

    with suppress(Exception):
        if os.name == 'posix':
            import shlex  # pylint:disable=import-outside-toplevel
            cmd = shlex.split(f"""sudo {sys.executable} -c 'import sys,spwd,crypt;"""
                              f"""                          ph = spwd.getspnam("{username}").sp_pwd;"""
                              f"""                          sys.exit(crypt.crypt("{password}",ph) != ph)'""")
            if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).returncode == 0:
                valid = None
        elif os.name == 'nt':
            import socket  # pylint:disable=import-outside-toplevel
            # noinspection PyUnresolvedReferences
            import win32security as wsec  # pylint:disable=import-outside-toplevel,import-error
            domain = '.'.join(socket.getfqdn().split('.')[1:])
            wsec.LogonUser(username, domain, password, wsec.LOGON32_LOGON_NETWORK, wsec.LOGON32_PROVIDER_DEFAULT)
            valid = None
    return valid


# ---------------------------
if __name__ == '__main__':
    print(password_validate_local(username=None, password=password_plaintext(os.getenv('PASSWORD', ''))))
