#!/usr/bin/env python3
# -*- mode: python -*-
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

""" setuptools/pip installer for cinch-pyutils. """

import os
from pathlib import Path

THISDIR = Path(__file__).resolve().parent
os.chdir(THISDIR)

import pyutils  # pylint:disable=wrong-import-position
from pyutils.setup import setup  # pylint:disable=wrong-import-position

setup(  # (dog food)
    THISDIR,
    version=pyutils.__version__,
    description="Generic Python Utility Functions for multipurpose development",
    url="https://github.com/cinchent/cinch-pyutils",
    author="Rod Pullmann",
    author_email='rod@cinchent.com',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: Eclipse Public License - v2.0",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Generic Python Utility Functions",
    ],
    keywords="python generic utilities",
    python_requires='>=3.6, <4',
    external_packages=False,
    executables=([__file__] + list(THISDIR.rglob('*.sh'))),
)
