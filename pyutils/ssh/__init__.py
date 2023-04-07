# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

""" SSH-related helper functions. """

import re
from pathlib import Path


def parse_ssh_config(filespec='~/.ssh/config', content=None):
    """
    Parses the SSH configuration file into a collection of hosts and their SSH characteristics.

    :param filespec: File specification for SSH config file (None => use canonical system default)
    :type  filespec: Union(str, None)
    :param content:  SSH configuration content (used intead of reading `filespec`)
    :type  content:  Union(str, None)

    :return: Hosts and their SSH access characteristics
    :rtype:  dict

    .. note::
     * A BATS host is distinguished by having a defined 'LocalCommand' definition in its configuration entry,
       which can be any non-empty text.  The current convention is to record the "rack name" location there,
       but that is not a hard requirement.
     * The SSH configuration file is presumed to be compatible with OpenSSH formatting, and is in the default
       user-specific location unless overridden explicitly via the 'SSH_CONFIG_FILE' envirosym.
    """
    if not content:
        try:
            content = Path(filespec).expanduser().read_text(encoding='utf-8')
        except FileNotFoundError:
            content = ''
    lines = [re.sub(r"^Host +", '\n\n', line.split('#')[0].strip(), flags=re.IGNORECASE)
             for line in content.split('\n')
             if line.strip() and not line.lstrip().startswith('#')]
    # noinspection PyTypeChecker
    hosts_text = dict(re.sub(r" +", ' ', line).split(';', maxsplit=1)
                      for line in ';'.join(lines).split('\n\n')[1:])
    hosts = {}
    for key, line in hosts_text.items():
        directives = [s.replace('=', ' ').split(maxsplit=1) for s in line.rstrip(';').split(';')]
        try:
            hosts[key] = dict(directives)
        except (Exception, BaseException):
            pass  # (no logging, just skip entry)

    return hosts
