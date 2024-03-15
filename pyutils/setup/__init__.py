# -*- mode: python -*-
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to support setuptools/pip installation.
"""
# pylint:disable=wrong-import-position

import sys
import os
import stat
from pathlib import Path
from contextlib import suppress
import shutil

from setuptools import (setup as _setup, find_packages)
from setuptools.command.install import install as _installer

sys.path.insert(0, str(Path(__file__).parents[2]))
# noinspection PyUnresolvedReferences,PyPackageRequirements
from pyutils.strings import truthy
# noinspection PyUnresolvedReferences,PyPackageRequirements
from pyutils.pip import (read_requirements, install_external_packages)
# noinspection PyUnresolvedReferences,PyPackageRequirements
from pyutils.git import GITHUB_DEFAULT_BRANCH


def parse_manifest(filespec):
    """
    Parses a setuptools MANIFEST.in file (required for sdist installs).

    :param filespec: File specification for MANIFEST.in file
    :type  filespec: Union(Path, str)

    :return: All package files denoted in MANIFEST.in file (in no particular order)
    :rtype:  list
    """
    manifest = set()
    lines = Path(filespec).read_text(encoding='utf-8').strip().split('\n')
    for line in lines:
        cmd, *params = line.split()
        if cmd == 'include':
            manifest |= set(params)
        elif cmd == 'recursive-include':
            manifest |= {"{}/{}".format(params[0], wc) for wc in params[1:]}
        elif cmd == 'graft':
            manifest |= {"{}/*".format(dirspec) for dirspec in params}
    return list(manifest)


def delete_junk(basedir):
    """
    Removes files/directories from the specified directory that should not be present
    for a clean installation.

    :param basedir: Directory specification for directory from which to remove junk
    :type  basedir: Union(Path, str)
    """
    basedir = Path(basedir)
    for filespec in basedir.rglob('*.py[co]'):
        filespec.unlink()
    for dirspec in ('__pycache__', '.pytest_cache'):
        for filespec in basedir.rglob(dirspec):
            shutil.rmtree(filespec, ignore_errors=True)


def make_executable(filespec):
    """
    Makes specified file(s) executable.

    :param filespec: File specification or list of filespecs for files to make executable.
    :type  filespec: Union(Path, str, list)
    """
    specs = (filespec,) if isinstance(filespec, (str, Path)) else filespec
    for spec in specs:
        try:
            path = Path(spec)
            if not path.exists():
                continue
            mode = os.stat(path).st_mode
            os.chmod(Path(spec),
                     mode
                     | (stat.S_IXUSR if mode & stat.S_IRUSR else 0)
                     | (stat.S_IXGRP if mode & stat.S_IRGRP else 0)
                     | (stat.S_IXOTH if mode & stat.S_IROTH else 0))
        except (Exception, BaseException):
            pass


# pylint:disable=expression-not-assigned
# noinspection PyTypeChecker
def setup(package_dir, requirements='.', supplemental_packages=(), external_packages=False, executables=(), **params):
    """
    Canonical wrapper for setuptools.setup().

    :param package_dir:           Package root directory
    :type  package_dir:           Union(str, Path)
    :param requirements:          File name or collection of files (names or Paths) for package dependencies
                                  (presumed relative to package root directory)
    :type  requirements:          Iterable
    :param supplemental_packages: Additional Python packages to find for installation (ignored if `packages` specified)
    :type  supplemental_packages: Iterable
    :param external_packages:     "Install external packages defined in `requirements`."
                                  (subject to envirosym definitions -- see notes)
    :type  external_packages:     bool
    :param executables:           Scripts/files to manually change permissions on to executable post-installation
    :type  executables:           Iterable
    :param params:                Additional options to be passed through directly to setuptools.setup()
    :type  params:                dict

    .. note::
     * Envirosyms heeded:
        - EXT_PACKAGES_REINSTALL => "(Re)install all external (non-PyPI/GitHub-resident) dependency packages."
                                    (denoted by ## annotations in requirements.txt)
        - EXT_PACKAGES_OVERWRITE => "Erase/overwrite external dependency package directories."
                                    (else safeguard existing package directories)
        - EXT_PACKAGES_DIR: Local base directory where all external dependency packages are installed into from GitHub
                            (otherwise, Python `site` directories are used) as pip "editable" packages
        - EXT_PACKAGES_AUTH: GitHub authentication scheme ('https' | 'ssh' | 'scp')
        - EXT_PACKAGES_CREDS: colon-delimited username:password/PAT to use for GitHub credentials
                              (see :function:`pyutils.git.url_reformat()` for how to use credential defaults)

    """
    package_dir = Path(package_dir).expanduser().resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory specification `{package_dir}' is not a directory")
    os.chdir(package_dir)  # (necessary for find_packages())

    # Specify defaults.
    url = params.get('url')
    with suppress(Exception):
        # noinspection PyUnresolvedReferences
        params.setdefault('download_url', url + f"/archive/{GITHUB_DEFAULT_BRANCH}.zip")
    with suppress(Exception):
        params.setdefault('name', Path(url).stem if url else package_dir.stem)
    params['name'] = params.get('name', '').replace('_', '-')
    with suppress(Exception):
        params.setdefault('long_description', package_dir.joinpath('README.md').read_text(encoding='utf-8'))
    params.setdefault('long_description_content_type', 'text/markdown')
    with suppress(Exception):
        params.setdefault('license', "Proprietary :: " +
                                     package_dir.joinpath('LICENSE.txt').read_text(encoding='utf-8').split('\n')[0])
    params.setdefault('packages', find_packages() + list(supplemental_packages)),
    params.setdefault('include_package_data', True)
    with suppress(Exception):
        params.setdefault('package_data', {'': list(parse_manifest(package_dir.joinpath('MANIFEST.in')))})

    if isinstance(requirements, str):
        requirements = (requirements,)
    requirements = [r if Path(r).is_absolute() else package_dir.joinpath(r) for r in requirements]
    with suppress(Exception):
        params.setdefault('install_requires', read_requirements(requirements, urls=False))
    params.setdefault('python_requires', '>=3.6, <4')
    params.setdefault('zip_safe', False)

    # Perform external package installation procedure.
    if external_packages:
        ext_packages_dir = os.getenv('EXT_PACKAGES_DIR')
        ext_packages_auth = os.getenv('EXT_PACKAGES_AUTH')
        ext_packages_creds = os.getenv('EXT_PACKAGES_CREDS')
        ext_package_urls = read_requirements(requirements, external=False, internal=False, urls=True,
                                             scheme=ext_packages_auth, creds=ext_packages_creds)
        ext_reinstall = truthy(os.getenv('EXT_PACKAGES_REINSTALL', True))  # pylint:disable=invalid-envvar-default
        ext_overwrite = truthy(os.getenv('EXT_PACKAGES_OVERWRITE', False))  # pylint:disable=invalid-envvar-default

        def install():
            install_external_packages(ext_package_urls, base_dir=ext_packages_dir, reinstall=ext_reinstall,
                                      overwrite=ext_overwrite, scheme=ext_packages_auth, creds=ext_packages_creds)

        class Installer(_installer):
            """ pip installer hook """
            def run(self):
                if ext_package_urls:
                    install()
                super().run()

        params['cmdclass'] = dict(install=Installer)

        if 'egg_info' in sys.argv:
            install()

    # Eliminate extraneous files.
    delete_junk(package_dir)

    # Perform the package installation.
    os.chdir(package_dir)  # (necessary for setup() and make_executables())
    _setup(**params)

    # Correct permissions for executables dynamically.
    if executables:
        make_executable(executables)
