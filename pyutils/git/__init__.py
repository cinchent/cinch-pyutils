# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2020-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide special functions for `git` and GitHub repositories.
"""
# pylint:disable=wrong-import-position,cyclic-import

import sys
import os
from pathlib import Path
from io import BytesIO
from urllib.parse import (urlparse, urlunparse)
from http import HTTPStatus
from base64 import b64decode
from functools import lru_cache
import subprocess
import shutil
import shlex
import zipfile

try:
    from git import Repo
except ImportError:
    Repo = None

try:
    import getpass
    # noinspection PyPackageRequirements
    import requests
except ImportError:  # (tolerate during 'pyutils' install only)
    getpass = requests = None

sys.path.insert(0, str(Path(__file__).parents[2]))
# noinspection PyUnresolvedReferences,PyPackageRequirements
from pyutils.cryptutils import (TextObfuscator, OBFUSCATOR_FILE, OBFUSKEY)

# --- GitHub Enterprise:
GHE = 'Cloud'
# ------- GitHub Enterprise Cloud (GHEC):
if GHE == 'Cloud':
    GITHUB_ENTERPRISE_DOMAIN = os.getenv('GITHUB_ENTERPRISE_DOMAIN', 'github.com')
    GITHUB_ENTERPRISE_DOMAIN_API = 'api.'  # GHEC domain prefix for API
    GITHUB_ENTERPRISE_API_BASE_PATH = '/'  # GHEC URL base path
    GITHUB_ENTERPRISE_ACCEPT_HEADER = 'application/vnd.github+json'     # HTTP request Accept header
# ------- GitHub Enterprise Server (GHES):
else:
    GITHUB_ENTERPRISE_DOMAIN = os.getenv('GITHUB_ENTERPRISE_DOMAIN', 'github.comcast.com')
    GITHUB_ENTERPRISE_DOMAIN_API = ''  # GHES domain prefix for API
    GITHUB_ENTERPRISE_API_BASE_PATH = '/api/v3/'  # GHES URL base path
    GITHUB_ENTERPRISE_ACCEPT_HEADER = 'application/vnd.github.v3+json'  # HTTP request Accept header

GITHUB_ENTERPRISE_ORG = os.getenv('GITHUB_ENTERPRISE_ORG', 'cinchent')  # GitHub Enterprise (GHE) org
# --- Common/Local:
GITHUB_TOKEN_FILE = os.getenv('GITHUB_TOKEN_FILE', '~/.github-token')   # File with GHE Personal Access Token (PAT)
GITHUB_TOKEN_ENVIROSYM = 'GITHUB_TOKEN'                                 # Envirosym specifying canonical GHE PAT
GITHUB_CACHES = os.getenv('GITHUB_CACHES', "pip")                       # Default repo caches
#                                                                         remote example: ssh://example.com/home/user
GITHUB_OBJECT_TYPES = ('file', 'dir', 'submodule')                      # Repo content retrieval types supported
GITHUB_URL_SCHEMES = ('http', 'https', 'ssh')                           # Supported GitHub URL schemes
GITHUB_URL_SCHEMES_DEPRECATED = ('git', 'ftp', 'ftps')                  # Deprecated GitHub URL schemes
GITHUB_DEFAULT_USERNAME = 'git'                                         # Default GitHub user
GITHUB_DEFAULT_REPO_SUFFIX = 'git'                                      # Default GitHub repo URL path suffix
GITHUB_DEFAULT_BRANCH = 'main'                                          # Default GitHub repo branch

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


# pylint:disable=protected-access
# noinspection PyProtectedMember
def url_reformat(url, scheme=None, creds=None, suffix=None, ref=None):  # noqa:C901
    """
    Reformats a GitHub repository URL to a canonical format of the specified URL scheme, optionally including
    credentials and a "commit-ish" reference specification.

    :param url:     Repository URL
    :type  url:     str
    :param scheme:  Schema to reformat input URL to:
                      'http[s]:' => standard URL format using http[s] scheme
                      'ssh' => standard URL format using ssh scheme
                      'scp' => no scheme, GitHub SSH "scp-like" format (i.e., `username`@`netloc`:`org`/`repo_path`)
                      None => use scheme, if any, as specified in input URL (not validated)
    :type  scheme:  Union(str, None)
    :param creds:   Authorization username and/or password/PAT for GitHub server -- tuple or `;`-separated string,
                    where tuple as (`username`, [`password_or_PAT`]), or string as:
                      `username` => credentials have no password/PAT
                      `username:` => use canonical PAT for credentials
                      `:password_or_PAT` => use current username for credentials
                      `:` => use current username and canonical PAT for credentials
                      empty => remove credentials entirely (use 'git' for `scheme`=='scp') in output URL
                      None => use creds, if any, as specified in input URL
    :type  creds:   Union(Iterable, None)
    :param suffix:  Suffix for URL path:
                      str => use as suffix
                      '.' => use default GitHub suffix
                      empty => remove suffix from input URL
                      None => use suffix, if any, as specified in input URL
    :type  suffix:  Union(str, None)
    :param ref:     Reference specification (i.e., branch/tag/commit) appended to URL with `#` delimiter:
                      str => use as refspec
                      empty => remove refspec from input URL
                      None => use refspec, if any, as specified in input URL
    :type  ref:     Union(string, None)

    :return: Result tuple:
              [0]: Reformatted URL (None => error: ill-formed input URL or param specification)
              [1]: Parsed URL parts (all components empty in case of error)
    :rtype:  tuple

    .. note::
     * See https://git-scm.com/docs/git-clone#_git_urls for supported git URL syntax.
     * Deprecated schemas are still recognized as valid in the input URL, but will not be produced as an output URL
       schema unless the `schema` parameter is specified as `None` (thus conserving the input schema).
     * The prefixed `git+xxx` schema is recognized and handled in the input URL as if the prefix weren't present,
       with the prefix prepended onto the output URL.
     * URLs that include an explicit port number as part of the server netloc are accommodated, but will be ambiguous
       when reformatting to an "scp-like" output URL.
     * If a "canonical" PAT is used for GitHub credentials, it is retrieved from the envirosym identified by the
       GITHUB_TOKEN_ENVIROSYM definition above, or failing that, from the (possibly obfuscated) text file as denoted
       by the GITHUB_TOKEN_FILE definition above.
    """
    parts = urlparse(url)
    try:
        scheme_sep = '://'
        prefix_sep = '+'

        # Special case: "file" scheme URL is passed through unmodified unless scheme change is erroneously attempted.
        if parts.scheme == 'file':
            raise (StopIteration() if not scheme or scheme == 'file' else TypeError("cannot reformat to file scheme"))

        # Standardize specified URL components and validate against deprecated output schemes.
        if not parts.netloc:  # (special case: parsing failed for "scp-like" scheme)
            parts = urlparse(scheme_sep.join(('scp', url)))  # (use "scp" as a pseudo-scheme to correct parsing)
            parts = parts._replace(**dict(zip(('netloc', 'path'), (parts.netloc + parts.path).rsplit(':', maxsplit=1))))

        # Determine scheme prefix.
        prefix, unprefixed_scheme = (prefix_sep + (scheme or parts.scheme)).rsplit(prefix_sep, maxsplit=1)
        prefix = (prefix + prefix_sep).lstrip(prefix_sep)
        if scheme and not scheme.startswith(prefix):  # (ignore prefix unless conserving scheme)
            prefix = ''
        if prefix and parts.scheme.startswith(prefix):
            parts = parts._replace(scheme=unprefixed_scheme)

        if scheme:
            scheme = unprefixed_scheme.lower()
            if scheme in GITHUB_URL_SCHEMES_DEPRECATED:
                raise TypeError("deprecated scheme")
        else:
            scheme = parts.scheme

        # Extract and/or substitute defaults for credentials:
        orig_creds, netloc = ('@' + parts.netloc).rsplit('@', maxsplit=1)
        if creds is None:
            creds = orig_creds.lstrip('@')
            if scheme in ('http', 'https') and creds == GITHUB_DEFAULT_USERNAME:
                creds = ''
        elif creds:
            if not isinstance(creds, str):
                creds = ':'.join(creds)
            if ':' in creds:
                username, password = creds.split(':', maxsplit=1)
                creds = ':'.join((username.strip() or getpass.getuser(), password.strip() or get_token()))
        if scheme not in ('http', 'https') and not creds:
            creds = GITHUB_DEFAULT_USERNAME  # automatically use default username unless reformatting to http[s] scheme

        netloc = '@'.join((creds, parts.netloc.split('@')[-1])[not creds:])

        # Determine reference specification for output URL.
        if ref is None:
            ref = parts.fragment

        # Add/remove/alter path suffix.
        if suffix is not None:
            suffix = ('.' + GITHUB_DEFAULT_REPO_SUFFIX if suffix == '.' else
                      '.'[not suffix or suffix.startswith('.'):] + suffix)
            parts = parts._replace(path=str(Path(parts.path).with_suffix(suffix)))

        # ---- Format SSH-like scheme:
        if scheme in ('ssh', 'scp'):
            parts = parts._replace(scheme=scheme.split('scp')[0], netloc=netloc,
                                   path='/:'[scheme == 'scp'] + parts.path.lstrip('/:'),
                                   fragment=ref)
            # noinspection PyTypeChecker
            url = urlunparse(parts).lstrip('/')
            if scheme == 'scp':  # (special case: eliminate path separator automatically inserted by unparse())
                # noinspection PyTypeChecker
                url = url.replace('/:', ':')

        # ---- Format HTTP-like scheme:
        elif scheme in ('http', 'https') + GITHUB_URL_SCHEMES_DEPRECATED:
            parts = parts._replace(scheme=scheme, netloc=netloc, fragment=ref)
            url = urlunparse(parts)

        # ---- (unknown)
        else:
            raise ValueError("unrecognized scheme")

        # Apply scheme prefix as applicable.
        # noinspection PyTypeChecker
        url = prefix + url
    except StopIteration:
        pass
    except (Exception, BaseException):  # pylint:disable=broad-except
        url, parts = None, urlparse('')

    return url, parts


# pylint:disable=invalid-name
def deploy_repo(base_dir, repo_name, url, branch=None, overwrite=False, try_sudo=False, **_):  # noqa:C901
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
    :param try_sudo:  "Try using 'sudo' to acquire repo if acquisition as user fails."
    :type  try_sudo:  bool

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
            ERROR_LOG(f"ERROR: Access error for repo base directory '{base_dir}': {exc}\nCannot proceed")
            ok = False
            break

        suffix = Path(urlparse(url).path).suffix
        if suffix not in ('.git', ''):
            raise NotImplementedError(f"Unsupported repo acquisition type: '{suffix}'")

        # Delete target installation directory if it exists (unless safeguarded).
        if not overwrite and Path(repo_name).exists():
            ERROR_LOG(f"WARNING: Repo target directory '{repo_name}' already exists, installing existing contents")
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
                if try_sudo:
                    subprocess.run("sudo rm -rf {}".format(repo_name), shell=True, check=True)
                else:
                    raise
            except (Exception, BaseException) as exc:
                err = str(exc)
        except (Exception, BaseException) as exc:
            err = str(exc)
        if err:
            ok = False
            ERROR_LOG("ERROR: Failure deleting existing repo directory '{}': {}"
                      .format(Path(repo_name).resolve().absolute(), err))
            break

        # Clone the package repo/branch.
        if branch:
            branch = f"--branch {branch}"
        for pfx in ('', 'sudo')[:1 + try_sudo]:
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
                ERROR_LOG(f"WARNING: Failure installing repo '{repo_name}' as user: {err}\nTrying as root...")
            except (Exception, BaseException) as exc:
                err = str(exc)
        if err:
            ok = False
            ERROR_LOG(f"ERROR: Failure acquiring GitHub repo URL '{url}': {err}")
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
            # noinspection PyUnresolvedReferences,PyPackageRequirements
            from pyutils.pip import get_package_dir  # pylint:disable=import-outside-toplevel
            foundpath = get_package_dir(reponame) if caches else None
            fulltree = False

        else:  # (try from specified URL)
            url_parts = urlparse(cache)
            if url_parts.scheme == 'file':  # (HTTP file URL scheme)
                foundpath = Path(url_parts.netloc).expanduser().joinpath(url_parts.path[bool(url_parts.netloc):])
                if not foundpath.is_absolute():
                    foundpath = None
                fulltree = False

            else:  # (all other HTTP URL schemes)
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


def git_repo(path):
    """ Accesses a locally cloned repo at a specific path. """
    if not Repo:
        raise NotImplementedError("GitPython package not installed")
    # noinspection PyCallingNonCallable
    return Repo(path)


def git_repo_files(repo, file_type=None):
    """ Retrieves the list of files of a particular type in a repo directory. """
    if not Repo:
        raise NotImplementedError("GitPython package not installed")
    if file_type == 'untracked':
        files = repo.untracked_files
    else:
        files = repo.git.ls_files().split('\n')
    return files


if __name__ == '__main__':
    # # Unit test-ish:
    this_repo = 'cinch-pyutils'
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

    # _valid = True
    # _valid &= (url_reformat("ssh://github.com/org/repo.git")[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://github.com/org/repo.git", scheme='https')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://github.com/org/repo.git", scheme='https', creds='me:secret')[0] ==
    #            'https://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://git@github.com/org/repo.git", scheme='https')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://git@github.com/org/repo.git", scheme='https', creds='me')[0] ==
    #            'https://me@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://git@github.com/org/repo.git", scheme='https', creds='me:')[0] ==
    #            f'https://me:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://git@github.com/org/repo.git", scheme='https', creds=':secret')[0] ==
    #            f'https://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:other_secret@github.com/org/repo.git", scheme='https', creds=':secret')[0] ==
    #            f'https://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:other_secret@github.com/org/repo.git", scheme='https', creds=':')[0] ==
    #            f'https://{getpass.getuser()}:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git", scheme='https', creds='other_me')[0] ==
    #            'https://other_me@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git", scheme='https', creds='')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git", scheme='https', creds=None)[0] ==
    #            'https://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git", scheme='https', ref='main')[0] ==
    #            'https://me:secret@github.com/org/repo.git#main')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git#feature", scheme='https', ref='main')[0] ==
    #            'https://me:secret@github.com/org/repo.git#main')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git#feature", scheme='https', ref=None)[0] ==
    #            'https://me:secret@github.com/org/repo.git#feature')
    # _valid &= (url_reformat("ssh://me:secret@github.com/org/repo.git#feature", scheme='https', ref='')[0] ==
    #            'https://me:secret@github.com/org/repo.git')
    # # ---
    # _valid &= (url_reformat("github.com:org/repo.git")[0] ==
    #            'git@github.com:org/repo.git')
    # _valid &= (url_reformat("github.com:org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("github.com:org/repo.git", scheme='ssh', creds='me:secret')[0] ==
    #            'ssh://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me@github.com:org/repo.git", scheme='ssh')[0] ==
    #            'ssh://me@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='ssh', creds='me')[0] ==
    #            'ssh://me@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='ssh', creds='me:')[0] ==
    #            f'ssh://me:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='ssh', creds=':secret')[0] ==
    #            f'ssh://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:other_secret@github.com:org/repo.git", scheme='ssh', creds=':secret')[0] ==
    #            f'ssh://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:other_secret@github.com:org/repo.git", scheme='ssh', creds=':')[0] ==
    #            f'ssh://{getpass.getuser()}:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='ssh', creds='other_me')[0] ==
    #            'ssh://other_me@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='ssh', creds='')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='ssh', creds=None)[0] ==
    #            'ssh://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git#feature", scheme='ssh', ref='main')[0] ==
    #            'ssh://me:secret@github.com/org/repo.git#main')
    # # ---
    # _valid &= (url_reformat("github.com:org/repo.git", scheme='https')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("github.com:org/repo.git", scheme='https', creds='me:secret')[0] ==
    #            'https://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me@github.com:org/repo.git", scheme='https')[0] ==
    #            'https://me@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='https')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='https', creds='me')[0] ==
    #            'https://me@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='https', creds='me:')[0] ==
    #            f'https://me:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("git@github.com:org/repo.git", scheme='https', creds=':secret')[0] ==
    #            f'https://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:other_secret@github.com:org/repo.git", scheme='https', creds=':secret')[0] ==
    #            f'https://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:other_secret@github.com:org/repo.git", scheme='https', creds=':')[0] ==
    #            f'https://{getpass.getuser()}:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='https', creds='other_me')[0] ==
    #            'https://other_me@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='https', creds='')[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git", scheme='https', creds=None)[0] ==
    #            'https://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("me:secret@github.com:org/repo.git#feature", scheme='https', ref='main')[0] ==
    #            'https://me:secret@github.com/org/repo.git#main')
    # # ---
    # _valid &= (url_reformat("https://github.com/org/repo.git")[0] ==
    #            'https://github.com/org/repo.git')
    # _valid &= (url_reformat("https://github.com/org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("https://github.com/org/repo", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo')
    # _valid &= (url_reformat("https://github.com/org/repo.git", scheme='ssh', creds='me:secret')[0] ==
    #            'ssh://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='ssh', creds='me')[0] ==
    #            'ssh://me@github.com/org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='ssh', creds='me:')[0] ==
    #            f'ssh://me:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='ssh', creds=':secret')[0] ==
    #            f'ssh://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:other_secret@github.com/org/repo.git", scheme='ssh', creds=':secret')[0] ==
    #            f'ssh://{getpass.getuser()}:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:other_secret@github.com/org/repo.git", scheme='ssh', creds=':')[0] ==
    #            f'ssh://{getpass.getuser()}:{get_token()}@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='ssh', creds='other_me')[0] ==
    #            'ssh://other_me@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='ssh', creds='')[0] ==
    #            'ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='ssh', creds=None)[0] ==
    #            'ssh://me:secret@github.com/org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git#feature", scheme='ssh', ref='main')[0] ==
    #            'ssh://me:secret@github.com/org/repo.git#main')
    # # ---
    # _valid &= (url_reformat("https://github.com/org/repo.git", scheme='scp')[0] ==
    #            'git@github.com:org/repo.git')
    # _valid &= (url_reformat("https://github.com/org/repo.git", scheme='scp', creds='me:secret')[0] ==
    #            'me:secret@github.com:org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='scp')[0] ==
    #            'git@github.com:org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='scp', creds='me')[0] ==
    #            'me@github.com:org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='scp', creds='me:')[0] ==
    #            f'me:{get_token()}@github.com:org/repo.git')
    # _valid &= (url_reformat("https://git@github.com/org/repo.git", scheme='scp', creds=':secret')[0] ==
    #            f'{getpass.getuser()}:secret@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:other_secret@github.com/org/repo.git", scheme='scp', creds=':secret')[0] ==
    #            f'{getpass.getuser()}:secret@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:other_secret@github.com/org/repo.git", scheme='scp', creds=':')[0] ==
    #            f'{getpass.getuser()}:{get_token()}@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='scp', creds='other_me')[0] ==
    #            'other_me@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='scp', creds='')[0] ==
    #            'git@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git", scheme='scp', creds=None)[0] ==
    #            'me:secret@github.com:org/repo.git')
    # _valid &= (url_reformat("https://me:secret@github.com/org/repo.git#feature", scheme='scp', ref='main')[0] ==
    #            'me:secret@github.com:org/repo.git#main')
    # # ---
    # _valid &= (url_reformat("http://github.com/org/repo.git", suffix='')[0] ==
    #            'http://github.com/org/repo')
    # _valid &= (url_reformat("http://github.com/org/repo", suffix='.git')[0] ==
    #            'http://github.com/org/repo.git')
    # _valid &= (url_reformat("http://github.com/org/repo.snarge", suffix='git')[0] ==
    #            'http://github.com/org/repo.git')
    # _valid &= (url_reformat("http://github.com/org/repo", suffix='.')[0] ==
    #            'http://github.com/org/repo.git')
    # _valid &= (url_reformat("http://github.com/org/repo.snarge", suffix='.')[0] ==
    #            'http://github.com/org/repo.git')
    # _valid &= (url_reformat("http://github.com/org/repo", scheme='ssh')[0] ==
    #            'ssh://git@github.com/org/repo')
    # _valid &= (url_reformat("https://github.com:443/org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com:443/org/repo.git')
    # _valid &= (url_reformat("https://github.com:443/org/repo.git", scheme='scp')[0] ==
    #            'git@github.com:443:org/repo.git')
    # _valid &= (url_reformat("ssh://github.com:22/org/repo.git", scheme='https')[0] ==
    #            'https://github.com:22/org/repo.git')
    # _valid &= (url_reformat("file:///path/to/local/repo.git", scheme=None)[0] ==
    #            'file:///path/to/local/repo.git')
    # _valid &= (url_reformat("file:///path/to/local/repo.git", scheme='file', suffix='snarge')[0] ==
    #            'file:///path/to/local/repo.git')
    # _valid &= (url_reformat("git://github.com:8080/org/repo.git", scheme=None)[0] ==
    #            'git://git@github.com:8080/org/repo.git')
    # _valid &= (url_reformat("git+https://github.com:443/org/repo.git", scheme=None)[0] ==
    #            'git+https://github.com:443/org/repo.git')
    # _valid &= (url_reformat("git+ssh://github.com:22/org/repo.git", scheme=None)[0] ==
    #            'git+ssh://git@github.com:22/org/repo.git')
    # _valid &= (url_reformat("git+ssh://github.com:22/org/repo.git", scheme='ssh')[0] ==
    #            'ssh://git@github.com:22/org/repo.git')
    # _valid &= (url_reformat("http://github.com/org/repo.git", scheme='git+ssh')[0] ==
    #            'git+ssh://git@github.com/org/repo.git')
    # _valid &= (url_reformat("ssh://github.com/org/repo.git", scheme='git+http')[0] ==
    #            'git+http://github.com/org/repo.git')
    # _valid &= (url_reformat("me:my_pat@github.com:org/repo.git", scheme='git+https', suffix='', creds=':')[0] ==
    #            f'git+https://{getpass.getuser()}:{get_token()}@github.com/org/repo')
