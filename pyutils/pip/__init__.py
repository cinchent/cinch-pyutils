# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2020-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide special functions to supplement 'pip' deficiencies.
"""
# pylint:disable=cyclic-import

import sys
import os
import re
import site
from pathlib import Path
from contextlib import suppress
from functools import lru_cache
from io import StringIO

REQUIREMENTS_FILESPEC = 'requirements.txt'

try:
    import getpass
except ImportError:  # (tolerate during 'pyutils' install only)
    getpass = None

try:
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pip._vendor.packaging.version')
    # noinspection PyPackageRequirements,PyProtectedMember
    from pip._internal.cli.main import main as pip_main  # pylint:disable=protected-access
except ImportError:
    warnings = None
    try:
        # noinspection PyPackageRequirements
        import pip
        pip_main = pip.main  # pylint:disable=no-member
    except ImportError:
        pip = pip_main = None  # pylint:disable=invalid-name

from pyutils.git import (get_token, url_add_auth, url_parse, deploy_repo)  # pylint:disable=wrong-import-position

ERROR_LOG = lambda *a, **k: print(*a, file=sys.stderr, **k)  # noqa:E731


@lru_cache()
def get_package_dir(pkgname):
    """
    Returns the package directory of the specified PIP-installed package in the local system.

    :param pkgname: Name of Python package
    :type  pkgname: str

    :return: Directory path of local package installation directory (Non => not installed)
    :rtype:  Union(None, Path)
    """
    stds = sys.stdout, sys.stderr
    sys.stdout = stdout = StringIO()
    sys.stderr = StringIO()
    try:
        _ = pip_main(['show', pkgname, '--disable-pip-version-check', '--no-cache-dir'])
        location = re.search(r'\nLocation: *(.*)\n', stdout.getvalue())
        if location:
            location = location.group(1).strip()
    except (Exception, BaseException):
        location = None
    finally:
        sys.stdout, sys.stderr = stds
    return Path(location) if location else None


def read_requirements(filespec='.', external=False, internal=True, packages=False, urls=True, auth=None):
    """
    Reads pip requirements file(s), and extracts all requirement lines.

    :param filespec: File/directory specification for requirements file (default: {REQUIREMENTS_FILESPEC}), or
                     list of such filespecs/dirspecs to process in sequence
    :type  filespec: Union(Path, str, Iterable)
    :param external: "Read 'external' requirements (see below)."
    :type  external: bool
    :param internal: "Read 'internal' (PyPA/local) requirements."
    :type  internal: bool
    :param packages: "Return package names instead of full requirement specifications."
    :type  packages: bool
    :param urls:     "Return URLs for 'external' requirements before requirement specifications."
    :type  urls:     bool
    :param auth:     Authorization for access to external repos:
                       * 'ssh' => use GitHub SSH access
                       * Username and password/token (pair or colon-separated string) => Auth for GitHub http access
                       * None => use URL as specified, unmodified
    :type  auth:     Union(Iterable, None)

    :return: All package requirements
    :rtype:  list

    .. note::
     * "External" package requirements are those that needed to be downloaded from code repos, specified as
       a double-comment prefix ('##') in the requirements file, specified in GitHub SSH or HTTPS URL format.
     * External package URLs may have a "@<branch>" or "@<sha>" suffix to qualify the repo context to install from.
    """
    filespecs = (filespec,) if isinstance(filespec, (str, Path)) else filespec
    all_urls, all_specs, all_packages, ext_packages = ([] for _ in range(4))
    for _filespec in filespecs:  # pylint:disable=too-many-nested-blocks
        filepath = Path(_filespec)
        if filepath.is_dir():
            filepath = filepath.joinpath(REQUIREMENTS_FILESPEC)
        for line in filepath.expanduser().read_text(encoding='utf-8').split('\n'):
            spec = line.strip()
            if not spec:
                continue
            package = url = ''
            if spec.startswith('#'):
                if not spec.startswith('##'):
                    continue
                url, *spec = [spec.split('##', maxsplit=1)[-1].strip()]
                if url:
                    package = Path(url).stem
                    ext_packages.append(package)
                    if urls:
                        all_urls.append(url_add_auth(url, auth) if auth else url)
                    if not external:
                        package = ''
            else:
                package = spec.split()[0]
                is_ext = package in ext_packages
                if url or not ((internal and not is_ext) or (external and is_ext)):
                    package = spec = ''
            if packages:
                if package and package not in all_packages:
                    all_packages.append(package)
            elif spec and spec.split() not in (s.split() for s in all_specs):
                all_specs.append(spec)

    return all_urls + (all_packages if packages else all_specs)


# pylint:disable=invalid-name
def install_external_packages(package_urls, base_dir=None, reinstall=False, overwrite=True, auth=None):
    """
    Process externally-installed packages separately (workaround for pip dependency links obsolescence).

    :param package_urls: Collection of SSH-compatible logins or HTTP URLs identifying external packages to clone
                         from an external repo and install; suffixed optionally with repo branch name
                         (example: git@github.com:org/package.git@branch or https://github.com/dir/package.git@branch)
    :type  package_urls: Iterable
    :param base_dir:     Base directory where to clone external dependency packages from GitHub
                         (None => use Python `site-packages` directory)
    :type  base_dir:     Union(str, Path, None)
    :param reinstall:    "Remove and reinstall already installed packages."
    :type  reinstall:    bool
    :param overwrite:    "Overwrite any existing cloned external package directories."
    :type  overwrite:    bool
    :param auth:         Authorization for access to external repos:
                           * 'ssh' => use GitHub SSH access
                           * Username and password/token (pair or colon-separated string) => Auth for GitHub http access
                           * None => use URL as specified; for http(s) URLs, modified with GitHub auth (see note below)
    :type  auth:         Union(Iterable, None)

    :return: "Successfully installed packages (or no package installation needed)."
    :rtype:  bool

    .. note::
     * Compatible only with pip >= 20.x .
     * pip installation has the following characteristics and limitations:
        - External packages are installed before the PyPI/local packages.
        - External packages are installed wherever (in whatever site directory) PyPI/local packages are
        - Package installation directories are not "editable".
        - Each external package is first installed ignoring any version specifiers (because of pip limitations), and
          version specifiers are then subsequently respected only if the corresponding package is also listed with
          those specifiers among the PyPA/local packages, but are not uninstalled if they found at that time to be
          non-conformant to those specifiers.

     * External packages are installed as "editable" pip packages.
     * When `base_dir` is unspecified, git cloning of external package is attempted first into any of the `site`
       user directories, then in the `site` system directories, until successful.
     * When `auth` is unspecified, http(s) URLs will be altered to use the current username and the GitHub PAT
       as authorization, where the PAT is retrieved according to `pyutils.git.get_token()`; note that any specification
       of `auth` other than 'ssh' is insecure, as the git remote URLs will contain the PAT as plaintext.
    """
    pip_opts = ['--disable-pip-version-check', '--no-cache-dir']

    # Find all editable packages installed.
    stdout_prev = sys.stdout
    stdout = sys.stdout = StringIO()
    try:
        _ = pip_main(['list', '--format', 'columns', '--editable'] + pip_opts)
        installed = {Path(t[-1]).stem: t[-1]
                     for t in [d.split() for d in stdout.getvalue().strip().split('\n')][2:]}
    except (Exception, BaseException):
        installed = {}
    finally:
        sys.stdout = stdout_prev

    # Determine external packages to (re)install.
    reinstalls = {}
    if not auth:
        with suppress(Exception):
            auth = (getpass.getuser(), get_token())
    for pkgurl in package_urls:
        pkgname, pkgurl, branch = url_parse(pkgurl, auth=auth)
        if not auth:
            ERROR_LOG(f"WARNING: Unspecified GitHub auth, cannot install package '{pkgname}'")
        if reinstall or pkgname not in installed:
            reinstalls[pkgname] = (pkgurl, installed.get(pkgname), branch)

    # Determine where to install external packages.
    if not base_dir:
        base_dir = os.getenv('EXT_PACKAGES_DIR')  # @@@ deprecated -- remove when setup.py converted everywhere
    if not base_dir:
        base_dir = Path('.')
    if base_dir:
        base_dir = Path(base_dir).expanduser().resolve()
        os.chdir(base_dir)
    else:
        for site_path, pip_loc in ([(site.getusersitepackages(), ['--user'])] +
                                   [(p, []) for p in site.getsitepackages()]):
            site_path = Path(site_path)
            with suppress(OSError):
                os.chdir(site_path)
                fs = Path('__setup__.tmp')
                fs.write_text('', encoding='utf-8')
                fs.unlink()
                os.makedirs('ext', exist_ok=True)
                os.chdir('ext')
                pip_opts.extend(pip_loc)
                break
        else:
            ERROR_LOG("WARNING: No suitable directory found for installing external packages --"
                      " must install these manually: {}".format(reinstalls))
            reinstalls = []

    # Clone and (re)install packages.
    ok = True
    for pkgname, (pkgurl, pkgpath, branch) in reinstalls.items():
        pkgpath = Path(pkgpath) if pkgpath else base_dir.joinpath(pkgname)
        if not deploy_repo(pkgpath.parent, pkgpath.stem, pkgurl, branch=branch, overwrite=overwrite):
            ok = False

        # Install the cloned package (may fail because 'pip' doesn't detect a successful installation during run).
        stderr_prev = sys.stderr
        stderr = sys.stderr = StringIO()
        err = None
        try:
            status = pip_main(['install', '--editable', pkgname] + pip_opts)
            if status != 0:
                raise RuntimeError("Failure installing '{}'".format(pkgname))
        except (Exception, BaseException) as exc:
            err = exc
        finally:
            sys.stderr = stderr_prev
        if err:
            stderr_text = stderr.getvalue().strip()
            if stderr_text:
                err = str(err) + ':\n' + stderr_text
            ok = False
            ERROR_LOG(f"WARNING: Failure pip-installing package '{pkgurl}': {err}")

    return ok


if __name__ == '__main__':
    # # Unit test-ish:
    # from itertools import product
    # _reqfile = Path('~/Repo/ea_test_foundation/requirements.txt').expanduser()
    # for _combo in product(*[(False, True)] * 4):
    #     _params = dict(zip('urls packages internal external'.split(), _combo))
    #     print("----- {}:".format(' '.join([('='.join((k, str(v)))) for k, v in _params.items()])))
    #     _r = read_requirements(_reqfile, **_params)
    #     print(_r)
    _ = None
