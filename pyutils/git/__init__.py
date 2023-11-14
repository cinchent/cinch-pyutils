# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2020-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide special functions for 'git' and 'GitHub' repositories.
"""
# pylint:disable=cyclic-import

import sys
import os
from pathlib import Path
from io import BytesIO
from urllib.parse import (urlparse, urlunparse)
from collections import namedtuple
from http import HTTPStatus
from base64 import b64decode
from functools import lru_cache
import subprocess
import shutil
import shlex
import zipfile

try:
    import getpass
    # noinspection PyPackageRequirements
    import requests
except ImportError:  # (tolerate during 'pyutils' install only)
    getpass = requests = None

from pyutils.cryptutils import (TextObfuscator, OBFUSCATOR_FILE, OBFUSKEY)

# --- GitHub Enterprise Server (GHES):
# GITHUB_ENTERPRISE_DOMAIN = os.getenv('GITHUB_ENTERPRISE_DOMAIN', 'github.comcast.com')  # GHES domain
# GITHUB_ENTERPRISE_DOMAIN_API = ''  # GHES domain prefix for API
# GITHUB_ENTERPRISE_API_BASE_PATH = '/api/v3/'  # GHES URL base path
# GITHUB_ENTERPRISE_ACCEPT_HEADER = 'application/vnd.github.v3+json'  # GHES HTTP request Accept header
# --- GitHub Enterprise Cloud (GHEC):
GITHUB_ENTERPRISE_DOMAIN = os.getenv('GITHUB_ENTERPRISE_DOMAIN', 'github.com')  # GHEC domain
GITHUB_ENTERPRISE_DOMAIN_API = 'api.'  # GHEC domain prefix for API
GITHUB_ENTERPRISE_API_BASE_PATH = '/'  # GHEC URL base path
GITHUB_ENTERPRISE_ACCEPT_HEADER = 'application/vnd.github+json'  # GHEC HTTP request Accept header
# --- Common:
GITHUB_ENTERPRISE_ORG = os.getenv('GITHUB_ENTERPRISE_ORG', 'cinchent')          # GitHub Enterprise (GHE) org
GITHUB_TOKEN_FILE = os.getenv('GITHUB_TOKEN_FILE', '~/.github-token')  # File containing GHE Personal Access Token (PAT)
GITHUB_TOKEN_ENVIROSYM = 'GITHUB_TOKEN'                                # Envirosym specifying overriding GHE PAT
GITHUB_CACHES = os.getenv('GITHUB_CACHES', "pip")                      # Default repo caches
#                                                                        remote example: ssh://example.com/home/user
GITHUB_OBJECT_TYPES = ('file', 'dir', 'submodule')
GITHUB_DEFAULT_BRANCH = 'main'                                         # Default repo branch

ERROR_LOG = lambda *a, **k: print(*a, file=sys.stderr, **k)  # noqa:E731


@lru_cache()
def get_token():
    """
    Retrieves the GitHub Enterprise Personal Access Token (PAT) for this server/user.

    :return: Retrieved PAT (None => unknown)
    :rtype:  Union(str, None)
    """
    token = os.getenv(GITHUB_TOKEN_ENVIROSYM)
    if not token:
        try:
            token = Path(GITHUB_TOKEN_FILE).expanduser().resolve().read_text(encoding='utf-8').strip()
        except (Exception, BaseException):
            pass
    if token.startswith('~'):
        try:
            key = OBFUSKEY or Path(OBFUSCATOR_FILE).expanduser().read_text(encoding='utf-8').strip()
            token = TextObfuscator.decode_text(token[1:], key=key)
        except (Exception, BaseException):
            pass
    return token


def url_add_auth(url, auth):
    """
    Adds username and password/token to a GitHub repository URL.

    :param url:  Repository URL
    :type  url:  str
    :param auth: Authorization type or username and password/PAT for GitHub server: tuple or ':'-separated string
                 ('ssh' => use SSH)
    :type  auth: Iterable

    :return: Adjusted URL (None => error)
    :rtype:  Union(str, None)
    """
    if not isinstance(auth, str):
        auth = ':'.join(auth)
    if ':' in auth:
        username, password = auth.split(':', maxsplit=1)
        auth = ':'.join((username.strip() or getpass.getuser(), password.strip() or get_token()))
    try:
        parts = urlparse(url)
        if auth == 'ssh':
            url = urlunparse(parts._replace(scheme='ssh', netloc='@'.join(('git', parts.netloc))))
        else:
            url = urlunparse(parts._replace(netloc='@'.join((auth, parts.netloc))))
    except (Exception, BaseException):  # pylint:disable=broad-except
        # noinspection PyTypeChecker
        url = None
    return url


GitHubURL = namedtuple('GitHubURL', 'name path branch')


def url_parse(url, auth=None):
    """
    Parses a GitHub repository URL, optionally modifying it to add authorization info.

    :param url:  Repository URL
    :type  url:  str
    :param auth: Authorization type or username and password/PAT for GitHub server: tuple or ':'-separated string
                 ('ssh' => use SSH); None => no auth specified
    :type  auth: Union(Iterable, None)

    :return: Parsed URL result, optionally modified to incorporate authorization additions
    :rtype:  GitHubURL
    """
    parts = urlparse(url.lstrip('git@'))
    _, *branch = parts.path.rsplit('@', maxsplit=1)
    # noinspection PyTypeChecker
    repo_name = Path(parts.path.split('@')[0]).stem
    if branch:
        branch = branch[0]
        url = url.rstrip('@' + branch)
    if auth and '@' not in parts.netloc:
        url = url_add_auth(url, auth)
    return GitHubURL(repo_name, url, branch or None)


# pylint:disable=invalid-name
def deploy_repo(base_dir, repo_name, url, branch=None, overwrite=False, **_):  # noqa:C901
    """
    Acquires content of a repository in entirety from GitHub and deploys it on the local filesystem.

    :param base_dir:  Base directory where to deploy acquired repo
    :type  base_dir:  Union(str, Path)
    :param repo_name: Directory name for repo on local system
    :type  repo_name: str
    :param url:       Repository URL
    :type  url:       str
    :param branch:    Branch (or SHA) name to retrieve from GitHub (None => default branch)
    :type  branch:    Union(str, None)
    :param overwrite: "Erase existing directory and install afresh if it exists."
    :type  overwrite: bool

    :return: "Repository was deployed successfully."
    :rtype:  bool

    .. note::
     * "Acquiring" the repo from GitHub depends upon the URL suffix, but defaults to a `git clone` operation; cloning
       will include all submodules defined for the repo as well.
     * Acquisition will be attempted first as the current user, then as 'root' (if possible); in the latter case,
       file/directory ownership for the entire directory tree will be reassigned to the current user.
    """
    ok, err = True, ''
    for _ in (True,):  # (break-able scope)
        try:
            base_dir = Path(base_dir).expanduser().resolve()
            os.chdir(base_dir)
        except OSError as exc:
            ERROR_LOG(f"ERROR: Access error for repository base directory '{base_dir}': {exc}\nCannot proceed")
            ok = False
            break

        suffix = Path(urlparse(url).path).suffix
        if suffix not in ('.git', ''):
            raise NotImplementedError(f"Unsupported repository acquisition type: '{suffix}'")

        # Delete target installation directory if it exists (unless safeguarded).
        if not overwrite and Path(repo_name).exists():
            ERROR_LOG(f"ERROR: Repository target directory '{repo_name}' already exists, installing existing contents")
            ok = False
            break
        try:
            shutil.rmtree(repo_name, ignore_errors=False)
        except PermissionError:
            try:
                shutil.move(repo_name, repo_name + '.deleteme')
            except (Exception, BaseException) as exc:
                err = str(exc)
        except FileNotFoundError:
            pass
        except OSError:
            try:
                subprocess.run("sudo rm -rf {}".format(repo_name), shell=True, check=True)
            except (Exception, BaseException) as exc:
                err = str(exc)
        except (Exception, BaseException) as exc:
            err = str(exc)
        if err:
            ok = False
            ERROR_LOG("ERROR: Failure deleting existing repository directory '{}': {}"
                      .format(Path(repo_name).resolve().absolute(), err))
            break

        # Clone the package repo/branch.
        if branch:
            branch = f"--branch {branch}"
        for pfx in ('', 'sudo'):
            try:
                result = subprocess.run("{} git clone --recurse-submodules {} {} {}"
                                        .format(pfx, branch or '', url, repo_name),
                                        shell=True, check=False, encoding='utf-8',
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=200000)
                if result.returncode == 0:
                    if pfx == 'sudo':
                        subprocess.run("{0} chown -R {1}:{1} {2}".format(pfx, getpass.getuser(), repo_name),
                                       shell=True, check=True, bufsize=200000)
                    err = ''
                    break
                err = result.stdout.strip()
                ERROR_LOG(f"WARNING: Failure installing repository '{repo_name}' as user: {err}\nTrying as root...")
            except (Exception, BaseException) as exc:
                err = str(exc)
        if err:
            ok = False
            ERROR_LOG(f"ERROR: Failure acquiring GitHub repository URL '{url}': {err}")
    return ok


def get_repo_file(reponame, repopath, org=GITHUB_ENTERPRISE_ORG, ref=GITHUB_DEFAULT_BRANCH, caches=GITHUB_CACHES,
                  timeout=30):
    """
    Retrieves file content for a specific file from within a particular repository within a GitHub Enterprise (GHE)
    organization, optionally from a specific branch.

    :param reponame: Name of repo within GHE org
    :type  reponame: str
    :param repopath: POSIX-style path of file within repo (relative to repo root)
    :type  repopath: Union(str, Path)
    :param org:      GHE org where repo resides
    :type  org:      str
    :param ref:      Branch/tag/commit hash from which to retrieve file
    :type  ref:      str
    :param caches:   ;-delimited collection of repo cache URLs to search for file before accessing GHE
                     (file:// URL => cache is on local system; URL=='pip' => cache is installed Python package)
    :type  caches:   Union(str, None)
    :param timeout:  Timeout (sec) to retrieve file content
    :type  timeout:  float

    :return: Binary file content (must be decoded for ASCII text)
    :rtype:  bytes
    """
    contype, content, _ = _get_repo_content(reponame, repopath, repopath, org, ref, caches, timeout=timeout)
    if contype not in ('file', 'symlink'):
        raise TypeError(f"Content type mismatch: path identifies a '{contype}', not a file")
    return content


def get_repo_dir(reponame, repopath, org=GITHUB_ENTERPRISE_ORG, ref=GITHUB_DEFAULT_BRANCH, caches=GITHUB_CACHES,
                 maxdepth=1, flat=False, timeout=30):
    """
    Retrieves list of items for a specific directory (tree) from within a particular repository within a GitHub
    Enterprise (GHE) organization, optionally from a specific branch.

    :param reponame:  Name of repo within GHE org
    :type  reponame:  str
    :param repopath:  POSIX-style path of directory within repo (relative to repo root; empty => repo root)
    :type  repopath:  Union(str, Path)
    :param org:       GHE org where repo resides
    :type  org:       str
    :param ref:       Branch/tag/commit hash from which to retrieve directory
    :type  ref:       str
    :param caches:    ;-delimited collection of repo cache URLs to search for directory before accessing GHE
                      (file:// URL => cache is on local system; URL=='pip' => cache is installed Python package)
    :type  caches:    Union(str, None)
    :param maxdepth:  Max depth for recursion (1 => topmost level only, -1 => full depth)
    :type  maxdepth:  int
    :param flat:      "Compose flat list of full pathnames." (else hierarchical list-of-lists of file/dir names)
    :type  flat:      bool
    :param timeout:   Timeout (sec) to retrieve directory (tree)
    :type  timeout:   float

    :return: List of files/directories in directory
    :rtype:  list
    """
    repopath = Path(repopath) if repopath else ''

    def _get_repo_dir(_reponame, _repopath, _basepath, _org, _ref, _caches, _github, _maxdepth, _flat, _timeout):
        """ Retrieves specified directory from repo (NOTE: recursive). """
        contype, content, fulltree = _get_repo_content(_reponame, _repopath, _basepath, _org, _ref, _caches,
                                                       try_github=_github, maxdepth=_maxdepth, timeout=_timeout)
        if contype != 'dir':
            raise TypeError(f"Content type mismatch: '{_repopath}' identifies a '{contype}', not a dir")
        if _repopath == _basepath:  # (topmost level)
            _github = content and 'sha' in content[0]
            if _github:
                _caches = ''

        _maxdepth -= 1
        sep = os.path.sep
        dirlist = currlist = []
        dirmap = {'': dirlist}
        for subpath in content:  # pylint:disable=too-many-nested-blocks
            if fulltree:
                if flat:
                    path = subpath.get('path')
                    if dirlist and dirlist[-1].rstrip(sep) == str(Path(path).parent):
                        dirlist = dirlist[:-1]
                    dirlist.append(path)
                else:
                    name = subpath.get('name')
                    base = subpath.get('path').rsplit(name, maxsplit=1)[0]
                    baselist = dirmap.get(base, [])
                    if baselist is not currlist:
                        if subpath.get('type') == 'dir':
                            if not baselist:
                                parent = Path(base).name
                                if currlist and currlist[-1] == parent + sep:
                                    currlist[-1] = {parent: baselist}
                                dirmap[base] = baselist
                        currlist = baselist
                    currlist.append(name)
            else:
                if _maxdepth != 0 and subpath.get('type') == 'dir':
                    dirname = subpath.get('name')
                    if dirname.rstrip(sep) in ('.git', ):
                        continue
                    subpath = _repopath.joinpath(dirname) if _repopath else Path(dirname)
                    _, subdir = _get_repo_dir(_reponame, subpath, _basepath, _org, _ref, _caches, _github, _maxdepth,
                                              _flat, _timeout)
                    if _flat:
                        dirlist.extend([str(Path(subpath.name, d)) for d in subdir] if _caches else subdir)
                    else:
                        dirlist.append({subpath.name: subdir})
                else:
                    dirlist.append(subpath.get(('name', 'path')[_flat]))

        return contype, dirlist

    return _get_repo_dir(reponame, repopath, repopath, org, ref, caches, True, maxdepth, flat, timeout)[-1]


def _get_repo_content_cached(reponame, repopath, ref, caches, maxdepth=-1):
    """ Helper function: retrieves repo content from a cached repo location. """
    contype, content, fulltree = None, None, False
    pathname = repopath.as_posix() if repopath else ''
    sep = os.path.sep

    # Try each specified repo cache in succession.
    for cache in (caches or '').split(';'):
        if content:
            break
        if not cache:
            continue

        if cache == 'pip':  # (try among locally pip-installed Python packages)
            from pyutils.pip import get_package_dir  # pylint:disable=import-outside-toplevel
            foundpath = get_package_dir(reponame) if caches else None
            fulltree = False

        else:  # (try from specified URL)
            url_parts = urlparse(cache)
            if url_parts.scheme == 'file':  # (HTTP file URL schema)
                foundpath = Path(url_parts.netloc).expanduser().joinpath(url_parts.path[bool(url_parts.netloc):])
                if not foundpath.is_absolute():
                    foundpath = None
                fulltree = False

            else:  # (all other HTTP URL schemas)
                foundpath = None
                # noinspection PyProtectedMember
                url = urlunparse(url_parts._replace(path=Path(url_parts.path, reponame).as_posix()))
                try:
                    # Retrieve entire zipped content of path from repo (in lieu of a command to retrieve just names).
                    blob = subprocess.check_output(shlex.split("git archive --format=zip --remote={} {} {}"
                                                               .format(url, ref, repopath)))
                except (Exception, BaseException):
                    continue
                with zipfile.ZipFile(BytesIO(blob)) as bundle:
                    fulltree = True
                    namelist = bundle.namelist()
                    basepath = namelist[0]
                    if pathname in namelist:  # (file)
                        contype = GITHUB_OBJECT_TYPES[0]
                        content = bundle.read(pathname)
                    elif pathname + sep == basepath:  # (dir)
                        contype = GITHUB_OBJECT_TYPES[1]
                        content = []
                        for path in namelist[1:]:
                            dirpath = path.split(basepath, maxsplit=1)[-1]
                            dirs = dirpath.rstrip(sep).split(sep)
                            if maxdepth < 0 or len(dirs) <= maxdepth:
                                content.append(dict(type=GITHUB_OBJECT_TYPES[path.endswith(sep)], path=dirpath,
                                                    name=dirs[-1] + sep[not path.endswith(sep):]))
        if foundpath:  # (found on local filesystem)
            basepath = foundpath.joinpath(repopath) if repopath else foundpath
            if basepath.is_dir():
                contype = GITHUB_OBJECT_TYPES[1]
                content = []
                for path in basepath.glob('*'):
                    is_dir = path.is_dir()
                    content.append(dict(type=GITHUB_OBJECT_TYPES[is_dir], name=path.name + sep[:is_dir],
                                        path=str(basepath.joinpath(path.name).relative_to(path)) + sep[:is_dir]))
            elif basepath.exists():
                contype = GITHUB_OBJECT_TYPES[0]
                content = basepath.read_bytes()

    return contype, content, fulltree


# pylint:disable=too-many-arguments
def _get_repo_content(reponame, repopath, basepath, org, ref, caches, try_github=True, maxdepth=-1, timeout=30.):
    """ Helper function: retrieves content from a repo cache or remote GHE API endpoint. """
    pathname = repopath
    if repopath:
        repopath = Path(repopath)
        if repopath.is_absolute():
            raise ValueError("`repopath` must be expressed relative to repo tree, not an absolute path")
        pathname = repopath.as_posix()

    # Opportunistically first check all repo caches specified, in order, for content.
    contype, content, fulltree = _get_repo_content_cached(reponame, repopath, ref, caches, maxdepth=maxdepth)
    # (fulltree: "Entire (flattened) tree of repo returned." (otherwise only topmost repo path dir))

    # Try searching GitHub (unless told not to) if content was not found in any repo cache.
    if not content and try_github:
        url = (f"https://{GITHUB_ENTERPRISE_DOMAIN_API}{GITHUB_ENTERPRISE_DOMAIN}{GITHUB_ENTERPRISE_API_BASE_PATH}"
               f"repos/{org}/{reponame}/contents/{pathname}?ref={ref}")
        token = get_token() or ''
        resp = requests.get(url, headers=dict(Accept=GITHUB_ENTERPRISE_ACCEPT_HEADER, Authorization=f"token {token}"),
                            timeout=timeout)
        metadata = resp.json()
        if resp.status_code != HTTPStatus.OK:
            message = metadata.get('message', resp.text)
            raise requests.exceptions.HTTPError(f"Error {resp.status_code} retrieving from GitHub: {message}")
        contype = GITHUB_OBJECT_TYPES[1]

        def _denote_subrepo(_d):
            _d['name'] = '/' + _d.get('name', '')
            _d['path'] = '/' + _d.get('path', '')

        if isinstance(metadata, dict):
            if metadata.get('type') == GITHUB_OBJECT_TYPES[-1]:
                _denote_subrepo(metadata)
                content = [metadata]
            else:
                contype = GITHUB_OBJECT_TYPES[0]
                content = metadata.get('content', '')
                content = b64decode(content) if metadata.get('encoding') == 'base64' else content
        else:
            fulltree = False
            content = metadata
            for ent in content:
                ent['path'] = str(Path(ent.get('path')).relative_to(basepath))
                if ent.get('type') == GITHUB_OBJECT_TYPES[1]:
                    ent['name'] += os.path.sep
                    ent['path'] += os.path.sep
                elif not ent.get('download_url'):  # (kludge: correct for subrepo misidentified as file)
                    ent['type'] = GITHUB_OBJECT_TYPES[-1]
                    _denote_subrepo(ent)

    return contype, content, fulltree


if __name__ == '__main__':
    # # Unit test-ish:
    this_repo = 'cinchent-pyutils'
    # _token = get_token()
    # _dir = get_package_dir(this_repo)
    # _file = get_repo_file(this_repo, 'pyutils/git/__init__.py')
    # _file = get_repo_file(this_repo, Path('pyutils/git', '__init__.py'), caches=None)
    # _files = get_repo_dir(this_repo, 'pyutils/git', caches=None)
    # _files = get_repo_dir(this_repo, 'pyutils/git')
    # _files = get_repo_dir(this_repo, 'pyutils/git', caches=None)
    # _files = get_repo_dir(this_repo, 'pyutils/git', flat=True)
    # _files = get_repo_dir(this_repo, 'unittests', flat=True, maxdepth=-1)
    # _files = get_repo_dir(this_repo, 'unittests', flat=True, maxdepth=-1, caches=None)
    # _files = get_repo_dir(this_repo, '', maxdepth=-1)
    # _files = get_repo_dir(this_repo, '', maxdepth=-1, caches=None)
    # _files = get_repo_dir(this_repo, '', flat=True, maxdepth=-1)
    # _files = get_repo_dir(this_repo, '', flat=True, maxdepth=-1, caches=None)
    # _files = get_repo_dir(this_repo, '', maxdepth=2)
