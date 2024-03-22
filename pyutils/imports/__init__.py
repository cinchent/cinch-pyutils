# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2023  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

""" Importation helper functions. """
import site
import sys
import os
import re
from pathlib import Path
import builtins
import inspect
from collections import (Counter, namedtuple)
from types import SimpleNamespace
from contextlib import suppress
import operator
import subprocess
import json
from urllib.parse import urlparse
import tempfile
from textwrap import dedent
import importlib
import importlib.machinery as impmach
import importlib.util as imputil
import warnings

# noinspection PyDeprecation
import pkg_resources
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

try:
    from importlib.metadata import version as pkg_version
except ImportError:
    def pkg_version(pkg_name):
        """ Resolve installed package version (pre-importlib). """
        return pkg_resources.require(pkg_name)[0].version

try:
    import requirements
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from distutils.version import LooseVersion as Version  # pylint:disable=deprecated-module
    warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
except ImportError:
    requirements = None
    Version = None


# noinspection GrazieInspection
def add_sys_path(dirspec, prepend=False):
    """
    Adds a directory to the Python system path.

    :param dirspec: Directory specification to add to system path (may be
                    a pathlib.Path or anything convertible to a string)
    :param prepend: "Prepend the directory to the system path."
                    (else retains its existing position or is appended)
    :type  prepend: bool

    :return: Directory added (as specified)

    .. note::
     * The directory need not exist to be added.
     * Duplicate directories in the system path are removed
       as a side-effect of this function.
    """
    dirstr = str(dirspec)
    new_path = [dirstr] + sys.path if prepend else sys.path + [dirstr]
    sys.path = list(Counter(new_path).keys())
    return dirspec


def import_module(name, alias=None, within=None, cached=None):
    """
    (Re-)imports a module (relative to a package), optionally as an alias
    within another module, optionally replacing an existing module in the
    system module cache.

    :param name:   Name of module to import (qualified within system path)
    :type  name:   str
    :param alias:  Module name alias
    :type  alias:  Union[str, None]
    :param within: Namespace/object/dict or module name within which to assign
                   `alias` or `name` (None => import only, no assignment)
    :type  within: Union[Namespace, dict, str, None]
    :param cached: Fully-qualified module name (in sys.modules) to (re)assign
                   with imported module (None => default cache assignments)
    :type  cached: Union[str, None]

    :return: Imported module
    :rtype:  module

    .. note::
     * Import is always attempted, even if the module is already in the system
       module cache.
     * Only absolute imports are supported.
     * Uses `importlib.import_module()` for the actual importation, so is
       subject its limitations and whatever exceptions that may raise.
     * `import A as B` would be specified as: `B = import_module('A')` or
         `import_module(name='A', alias='B', within=globals())`
     * If `within` is specified as a module name, the namespace of that cached
       module (which must exist in the system cache) is used, and `alias`
       (if specified) or `name` (if not) is assigned in that namespace with
       the imported module object.
    """
    sys.modules.pop(name, None)
    module = importlib.import_module(name)
    if within is not None:
        if isinstance(within, str):
            within = sys.modules.get(within)
            if not within:
                raise ModuleNotFoundError("No module {} to import within"
                                          .format(within))
        alias = alias or name
        if isinstance(within, dict):
            within[alias] = module
        else:
            namespace = within
            *subspaces, alias = alias.rsplit('.', maxsplit=1)
            for subspace in subspaces:
                setattr(namespace, subspace, SimpleNamespace())
                namespace = getattr(namespace, subspace)
            setattr(namespace, alias, module)
        if cached:
            sys.modules[cached] = module
    return module


def import_module_source(modname, filespec=None, altpath=None, code=None, expand=True, execute=False):  # noqa:C901
    """
    Imports a module by source file, assuming it contains Python source code, irrespective of its file
    extension.

    :param modname:  Name of module under which to import source code
    :type  modname:  str
    :param filespec: File specification for file containing Python source (None => use `code`)
    :type  filespec: Union[str, None]
    :param altpath:  Directory specification of an alternate path to search for `filespec` (None => no alternate path)
    :type  altpath:  Union[str, None]
    :param code:     String containing source code to "import" as a module (None => import from `filespec`)
    :type  code:     Union[str, None]
    :param expand:   "In case of importation failure, inserts sourced file(s) inline and substitutes values
                      for all symbol references expressed in a subset of shell parameter expansion notation."
    :type  expand:   bool
    :param execute:  "In case of importation failure, executes the specified file as a shell script instead,
                      assigning all environment variables generated as module variables."
    :type  execute:  bool

    :return: Python source module
    :rtype:  module

    .. note::
     * When both `expand` and `execute` are specified, expansion is attempted before execution.
    """
    try:
        if filespec:
            syspath = sys.path
            if altpath:
                syspath.append(str(altpath))
        else:
            syspath = []
        modfilespec = filespec
        # noinspection PyTypeChecker
        for pathdir in [None] + syspath:
            error = None
            if modfilespec:
                if pathdir:
                    modfilespec = Path(pathdir, filespec)
                elif not Path(filespec).is_absolute():
                    continue
                modfilespec = modfilespec.resolve()
                loader = impmach.SourceFileLoader(modname, str(modfilespec))
                module = imputil.module_from_spec(imputil.spec_from_loader(modname, loader))
                try:
                    loader.exec_module(module)  # (load_module() deprecated)
                except FileNotFoundError:
                    continue
                except (Exception, BaseException) as exc:
                    error = exc
            elif code:
                spec = imputil.spec_from_loader(modname, None, origin='<string>')
                module = imputil.module_from_spec(spec)
                try:
                    # pylint:disable=exec-used
                    exec(code, vars(module))
                except (Exception, BaseException) as exc:
                    error = exc
            else:
                raise ImportError("'filespec' or 'code' must be specified")

            if type(error) in (SyntaxError, ImportError):  # (normal import failed)
                try:
                    if not expand:
                        raise error from error
                    define_script_vars(module, modfilespec, content=code)
                except (Exception, BaseException):  # (script expansion failed)
                    if not execute:
                        continue
                    try:
                        execute_script_vars(module, modfilespec, content=code)
                    except (Exception, BaseException):  # (script execution failed)
                        continue
            elif error:
                raise error from error
            sys.modules[modname] = module
            break
        else:  # (all paths exhausted)
            raise ImportError("Module file '{}' not importable".format(filespec))
    except (Exception, BaseException) as exc:
        raise ImportError(exc) from exc
    return module


def lookup_submodule(module, name):
    """ Looks up a submodule of a specified module by name. """
    name = name.split('.', maxsplit=1)[-1]
    if module:
        child, *subs = name.split('.', maxsplit=1)
        if module.__name__ != child:
            module = vars(module).get(child)
        if module and subs:
            module = lookup_submodule(module, name)
    return module


# pylint:disable=exec-used
def define_script_vars(obj, filespec, content=None):
    """
    Interprets a shell script containing variable definitions, expanding
    scripts referenced via `source` statements, performing shell variable
    substitutions, and assigning all variable definitions as object
    attributes.
    """
    if content:
        srcdir = ''
    else:
        content = filespec.read_text()
        srcdir = filespec.parent.resolve()
    symdefs = {}
    while True:
        symdef = re.search(r"""\n(\w+)=(?P<q>['"])(.*?)(?P=q)\s*\n""", content)
        if symdef:
            sym, _, val = symdef.groups()
            if re.match(r"""\$\(.*dirname.*\)""", val):
                val = str(srcdir)
                symdefs.update({sym: val})
            content = content.replace(symdef.group(0), """\n{}=r'''{}'''\n""".format(sym, val))
        source = re.search(r"""\n\s*source\s*(?P<q>['"])(.*?)(?P=q)\s*\n""", content)
        if source:
            try:
                srcspec = symsubst(source.group(2), defaults=symdefs, strict=True)
            except ValueError:
                exec(symsubst(content[:source.span()[0]], defaults=symdefs), symdefs)
                continue
            srcdir = Path(srcspec).parent
            content = content.replace(source.group(0), Path(srcspec).read_text(encoding='utf-8'))
        if not (source or symdef):
            break
    exec(content, obj.__dict__)  # (base symbol definitions, without substitutions)
    exec(symsubst(content, defaults={**symdefs, **obj.__dict__}), obj.__dict__)  # (substituted symbol definitions)


def execute_script_vars(obj, filespec, content=None):
    """
    Executes a shell script, assigning all environment variable definitions
    it produces as object attributes.
    """
    exec_fd, exec_filespec = tempfile.mkstemp(suffix='.sh')
    try:
        with os.fdopen(exec_fd, 'wt') as exec_file:
            source_code = content
            if not source_code:
                source_code = dedent("""\
                pushd {} >/dev/null
                set -a; source {} >/dev/null
                popd >/dev/null
                """).format(*os.path.split(str(filespec)))
            exec_file.write(dedent("""\
                #!/usr/bin/env bash
                {}
                env
                """).format(source_code))
        os.chmod(exec_filespec, 0o700)
        symdefs = subprocess.check_output([exec_filespec], shell=True, encoding='utf-8').split('\n')
        for symdef in symdefs:
            if not symdef.strip():
                continue
            name, value = symdef.split('=', maxsplit=1)
            if os.getenv(name, None) != value:
                with suppress(ValueError):
                    value = int(value)
                obj.__dict__[name] = value
    finally:
        os.unlink(exec_filespec)


def import_extended(modname, objects=(), paths=(), default=NotImplemented):
    """
    Imports a module (or objects from a module), optionally appending
    each of a collection of directories to the Python system path upon
    successive importation failures.

    :param modname:  Name of (sub)module to import
    :type  modname:  str
    :param objects: Collection of object names to import;
                    specified as str > single name
                    (equivalent to "from `modname` import `objects`";
                     empty => "import `modname`")
    :type  objects: Union[Iterable, str]
    :param paths:   Collection of directory specifications to successively
                    append to Python system path: each may be a pathlib.Path
                    or anything convertible to a string;
                    specified as str => treated as a single path
                    (empty => no alternate path locations)
    :type  paths:   Union[Iterable, str]
    :param default: On importation failure, any default value to use instead
                    of the module or each missing object (e.g., None or Mock)
                    (NotImplemented => propagate raised ImportError)

    :return: Imported module (if no objects specified), list of imported
             objects (if objects specified), or `default` (if import failed)

    .. raises:: ImportError if import fails and `default` unspecified
    .. note::
     * Always attempts to import; does not use Python system modules cache.
     * On successful module import, inserts module to the Python system modules
       cache dictionary by specified name.
     * Differs from import statement in that module/object names cannot be
       assigned automatically in globals; explicit assignment of function result
       is required to emulate this behavior.
    """
    result = default
    import_exc = None
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    for pathidx in range(-1, len(paths)):
        try:
            result = __import__(modname, fromlist=objects)
            break
        except ImportError as exc:
            if pathidx + 1 >= len(paths):
                continue
            add_sys_path(paths[pathidx + 1])
            import_exc = exc
    else:  # (all paths exhausted)
        if default is NotImplemented:
            raise import_exc

    if not objects:  # (import modname)
        if result is not default:
            sys.modules[modname] = result
    else:  # (from modname import objects)
        if isinstance(objects, str):
            objects = (objects,)
        absent = list(filter(lambda o: o not in vars(result), objects))
        if absent and default is NotImplemented:
            raise ImportError("Cannot import name{} '{}' from '{}' ({})"
                              .format('s'[:len(absent) > 1],
                                      ", ".join(absent),
                                      modname, result.__file__))
        objlist = [getattr(result, objname, default) for objname in objects]
        result = objlist[0] if len(objects) == 1 else objlist

    return result


def symsubst(text, environ=None, defaults=None, strict=False):  # noqa: C901
    """
    Performs symbolic expansion of a specified text string, substituting
    values for environment symbols or, alternatively, default symbol
    definitions, if any.

    :param text:     Unsubstituted text to process
    :type  text:     str
    :param environ:  Environment to apply (None => use shell environment)
    :type  environ:  Union[dict, None]
    :param defaults: Default symbol values to use if absent from `environ`
    :type  defaults: Union[dict, None]
    :param strict:   "Disallow substitutions from undefined symbols."
    :type  strict:   bool

    :return: Fully-substituted text
    :rtype:  str

    .. note::
     * All well-formed bash-style symbol references are expanded, using
       values from (in order):
        - `environ`
        - `defaults`
        - inline default value (using :- or :=)
        - empty string
     * Only := and :- standard default specifications are supported.
     * Default specification :== forces the symbol assignment to the
       explicit inline default value, ignoring `environ` and `defaults`.
     * CAUTION! Uses recursion to handle nested expansions.
    """
    if environ is None:
        environ = os.environ
    symdefs = {}

    # pylint:disable=too-many-return-statements,invalid-name
    def _subst_ref(_text, _level=0):
        if _level > 10:  # (bottomless recursion guard)
            return _text

        # Identify all symbolic substitution references.
        nonlocal symdefs
        patt_simple_subst = re.compile(r"^\${(\w+)}")
        patt_default_subst = re.compile(r"^(\w+)(:[-=]+)?(.*)?$")
        symrefs = re.findall(r"\${.*}+", _text)
        for symref in symrefs:
            prevtext = _text
            while '$' in symref:  # (handle when multiple substitutions matched)
                ref, op, default = symref, '', (None if strict else '')
                sym = re.match(patt_simple_subst, symref)
                if sym:      # (simple substitution)
                    ref = sym.group()
                    sym = sym.group(1)
                if not sym:  # (substitution with default operation)
                    sym, op, default = re.findall(patt_default_subst, symref[2:-1])[0]
                    if default and not op:
                        break  # (unsupported substitution operation)
                # Determine override value:
                symval = symdefs.get(sym, environ.get(sym, (defaults or {}).get(sym)))
                if not symval or symval == ref or op == ':==':
                    symval = default
                if strict and symval is None:
                    raise ValueError("Symbol unresolvable: '{}'".format(sym))
                symval = str(symval)

                # Perform any nested substitutions in the expansion text.
                if '$' in symval:
                    if any(s in symval for s in (f'${{{sym}}}', f'${{{sym}: '.rstrip())):
                        symdefs.setdefault(sym, '')
                    symdefs[sym] = symval = _subst_ref(symval, _level=_level + 1)

                # Substitute the text.
                _text = _text.replace(ref, symval)

                # Capture assignment, and assign target symbol in environment as a side-effect
                # when specified.
                symdefs.setdefault(sym, symval)
                if op.startswith(':='):
                    environ[sym] = symval

                # Skip to next expansion in this substitution (kludge needed
                # because outermost substitution extractions are greedy to
                # accommodate nesting).
                symref = symref[symref.find('$', len(ref)):]

                if _text == prevtext:  # (endless vacuous substitution guard)
                    break
                prevtext = _text
        return _text

    return _subst_ref(text)


def apply_environ(obj, environ=None, expandsyms=True, expanduser=True, prefix=''):  # noqa: C901
    # noinspection SpellCheckingInspection
    """
        Overrides each data attribute of an object with the corresponding
        environment definition.

        :param obj:        Object to process (must be a dict or have a __dict__ member)
        :param environ:    Environment snapshot to apply (None => use shell environment)
        :type  environ:    Union[dict, None]
        :param expandsyms: "Expand symbolic ${sym} and ${sym:-default} references."
        :type  expandsyms: bool
        :param expanduser: "Expand user home directory in (presumable) paths."
        :type  expanduser: bool
        :param prefix:     Environment symbol name prefix for all keys
        :type  prefix:     str

        :return: Object, with overridden data attributes

        .. note::
         * May be called recursively if dictionary subvalues are overridden.
         * Parses environment variable value as a space-/comma-delimited list
           of "words" if corresponding object value is listlike, preserving
           aggregate type for the object value during assignment.
        """
    if environ is None:
        environ = os.environ
    objdict = obj if isinstance(obj, dict) else vars(obj)
    for key, val in objdict.items():
        if key.startswith('__') or hasattr(val, '__dict__'):
            continue
        envkey = prefix + key
        substval = environ.get(envkey)
        if substval is None:
            substval = (apply_environ(val, environ=environ,
                                      expandsyms=expandsyms,
                                      expanduser=expanduser,
                                      prefix=envkey + '__')
                        if isinstance(val, dict) else val)
        else:
            with suppress(ValueError, TypeError):
                substval = int(substval)

        if isinstance(substval, str):
            if expandsyms:
                substval = symsubst(substval, environ=environ, defaults=objdict)
            if expanduser and "~" in substval:
                with suppress(Exception):
                    substval = str(Path(substval).expanduser())

        prevval = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        if isinstance(prevval, (tuple, list)):
            if isinstance(substval, (tuple, list)) and len(substval) == 1:
                substval = substval[0]
            if isinstance(substval, str):
                if substval and substval[0] in '[(':
                    substval = substval[1:-1]
                substval = substval.replace(',', ' ').split()
            substval = type(prevval)(substval)

        if isinstance(obj, dict):
            obj[key] = substval
        else:
            setattr(obj, key, substval)
    return obj


def update_environ(obj):
    """
    Updates ("exports") the process environment from the public attributes of
    a given object.

    :param obj: Object to export to environment (must have a __dict__ member)

    :return: Passed object

    .. note::
     * Private members are not exported
    """
    for key, val in vars(obj).items():
        if not key.startswith('_'):
            os.environ[key] = str(val)
    return obj


def isinstanceof(obj, cls, strict=False):
    """
    Fixes the native Python isinstance() function, allowing an object
    of a given class to be recognized as an instance of that class
    regardless of the import resolution technique for the module containing
    the class that happens to be used.

    :param obj:    Object (instance) of a given class
    :param cls:    Class to determine whether `obj` is an instance of
    :param strict: "Make comparison by examining source code."
                   (relatively intensive computationally)

    :return: "Object is an instance of the class."
    :rtype:  bool

    .. note::
     * (editorial) This fix is necessary due to the deficient Python import
       implementation, wherein two modules are not recognized as being identical
       if importation of one is resolved directly using 'sys.path' and
       importation of the other resolves the very same module via specification
       of the (sub)packages where the module resides -- the so-called
       "double-import trap"; the modules are regarded as distinct because their
       (sub)package affiliations differ.  By any objective standards, if these
       two techniques for how the import loader happens to resolve the modules
       result in both importing the class by name from the same source file,
       the classes should be considered as functionally interchangeable, and
       likewise for their instances; there is no use case where considering the
       classes as distinct is desirable.
     * This implementation devolves to the native Python built-in isinstance()
       function whenever possible, and digs deeper only when the native function
       fails to find a matching relationship.
     * In case a module may not be associated with any particular file or has no
       cached source code retrievable (e.g., in a "frozen" runtime environment),
       this method assumes a non-instance relation; although this is the case
       for built-in classes, the native isinstance() function will suffice to
       perform the comparison properly.
     * Warning: recursion used.
    """
    isinst = builtins.isinstance(obj, cls)
    if not isinst:  # (dig deeper)
        def _compare(_c1, _c2):
            _isinst = _c1.__name__ == _c2.__name__
            _isinst &= inspect.getfile(_c1) == inspect.getfile(_c2)
            if _isinst and strict:
                _isinst = (inspect.findsource(_c1)[-1] ==
                           inspect.findsource(_c2)[-1])
            return _isinst

        with suppress(Exception):
            isinst = (_compare(cls, obj.__class__) if isinstance(cls, type) else
                      any(isinstanceof(obj, c, strict=strict) for c in cls))
    return isinst


def package_name(name, package_naming=True):
    """
    Returns the specified type of name associated with a package.

    :param name:           Project name for a package
    :type  name:           str
    :param package_naming: "Return Python package name for a package."
                           (otherwise the project name)
    :type  package_naming: bool

    :return: Package/project name
    :rtype:  str

    .. note::
     * pip introduces the gratuitous abomination of "package name" vs.
       "project name" for each package: pip manages packages using the
       "package name" (*usually* dash-only word separators), whereas Python
       universally refers to package name (underscores as word-separators);
       `package_naming` allows the caller to specify the context of the
       naming scheme of interest for the result keys.
    """
    return name.replace(*('-', '_')[::+1 if package_naming else -1])


def parse_package_requirements(reqs, package_naming=True):
    """
    Parses pip "frozen" requirements as specified by text or file.

    :param reqs:           Requirements to parse
                           (starts with '@' => filespec of requirements file)
    :type  reqs:           Union[str, Path]
    :param package_naming: "Use package names, not project names for keys."
    :type  package_naming: bool

    :return: Collection of package metadata for each requirement; keys are
             project/package names, values are Requirement metadata
    :rtype:  dict<str, requirements.requirement.Requirement>

    .. note::
     * See `package_name()` for more about `package_naming`.
    """
    if not requirements:
        raise ModuleNotFoundError("'requirements-parser' package required")
    if isinstance(reqs, Path) or reqs.startswith('@'):
        reqs = (Path(str(reqs).lstrip('@')).expanduser()
                .read_text(encoding='utf-8'))
    return {package_name(p.name, package_naming=package_naming): p
            for p in requirements.parse(reqs)}


def check_package_requirements(reqs, packages=None, ignore=None,
                               package_naming=True):
    """
    Checks that all imported packages in specified collection of packages
    conform to the package requirements.

    :param reqs:           Requirements to parse
                           (starts with '@' => filespec of requirements file)
    :type  reqs:           Union[str, Path]
    :param packages:       Names of packages to check
                           (None => all packages in requirements file)
    :type  packages:       Union[Iterable, str, None]
    :param ignore:         Names of packages to ignore checking for
                           (None => all packages specified by `packages`)
    :type  ignore:         Union[Iterable, str, None]
    :param package_naming: "Use package names, not project names to refer
                            to packages."
    :type  package_naming: bool

    :return: Packages failing the requirements (empty => all ok) --
             each key: package name, each value is namedtuple:
               .actual:   actual version number (None => unknown)
               .required: list of requirements parsed from requirements file
                          (empty: package undefined in requirements file)
    :rtype:  dict<str, namedtuple>

    .. note::
     * `package_naming` determines the name format of `packages` and `ignore`
        (see note in `package_name()`)
    """
    if not requirements:
        raise ModuleNotFoundError("'requirements-parser' package required")
    if not Version:
        raise ModuleNotFoundError("'distutils.version' package required")

    package_mismatch = namedtuple('PackageMismatch', 'actual required')
    ops = {'==': '__eq__',
           '!=': '__ne__',
           '>':  '__gt__',
           '<':  '__lt__',
           '>=': '__ge__',
           '<=': '__le__',
           }

    reqlist = parse_package_requirements(reqs, package_naming=package_naming)

    if packages is None:
        packages = list(reqlist)
    elif isinstance(packages, str):
        packages = packages.split()

    if ignore is None:
        ignore = ()
    elif isinstance(ignore, str):
        ignore = ignore.split()

    failed = {}
    for pkgname in packages:
        if pkgname in ignore:
            continue
        req = reqlist.get(pkgname)
        req = req.specs if req else []
        try:
            actual = pkg_version(pkgname)
        except (Exception, BaseException):
            actual = None
        if not actual or not req:
            # noinspection PyArgumentList
            failed[pkgname] = package_mismatch(actual=actual, required=req)
            continue

        if not all(getattr(operator, ops.get(op), lambda *_: False)
                   (Version(actual), Version(ver))
                   for (op, ver) in req):
            # noinspection PyArgumentList
            failed[pkgname] = package_mismatch(actual=actual, required=req)

    return failed


def get_installed_packages(package_types='all', package_naming=True):
    """
    Retrieves all pip-installed packages of the specified type(s).

    :param package_types:  Type(s) of packages to include in result (list or
                           space-delimited string) -- any of:
                            * all      => All packages (supersedes all others)
                            * system   => "site" (all-user) packages
                            * user     => User-specific packages
                            * editable => Packages installed as "editable"
    :type  package_types:  Union[list, str]
    :param package_naming: "Use package names, not project names for keys."
    :type  package_naming: bool

    :return: Collection of metadata objects for all installed packages of the
             specified type(s), keys are "project name" of package metadata
    :rtype:  dict<str, pkg_resources.Distribution>

    .. note::
     * See `package_name()` for more about `package_naming`.
    """
    def _is_editable(_pkginfo, _sites):
        """ Utility: Determines if file location corresponding to a package indicates an "editable" package. """
        _editable = _pkginfo.location not in _sites
        if not _editable:
            with suppress(Exception):
                _egg_info = json.loads(Path(_pkginfo.egg_info).joinpath('direct_url.json').read_text(encoding='utf-8'))
                _editable = _egg_info.get('dir_info', {}).get('editable', False)
                if _editable:  # (if editable, convert entry to an EggInfoDistribution object)
                    _url = urlparse(_egg_info['url'])
                    _pkginfo.location = _url.path
        return _editable

    if isinstance(package_types, str):
        package_types = package_types.split()
    package_types = [p.lower() for p in package_types]
    sitepkgs = set()
    allsite = set(site.getsitepackages()) | {site.getusersitepackages()}
    if any(pt in package_types for pt in 'all system'.split()):
        sitepkgs |= set(site.getsitepackages())
    if any(pt in package_types for pt in 'all user'.split()):
        sitepkgs.add(site.getusersitepackages())
    editable = any(pt in package_types for pt in 'all editable'.split())
    # noinspection PyDeprecation
    working_set = pkg_resources.working_set.by_key
    return {package_name(p.project_name, package_naming=package_naming): p
            for p in working_set.values()
            if editable and _is_editable(p, allsite) or p.location in sitepkgs}


def build_dependencies(pkgname, dependencies=None, packages=None,
                       package_naming=True):
    """
    Updates an (optional) existing package dependencies DAG for the contribution
    due to a specified package, optionally limited to an applicable universe of
    installed packages.

    :param pkgname:        Name of package/project (see `package_naming`)
    :type  pkgname:        str
    :param dependencies:   Existing (known) dependencies, to which dependencies
                           for `pkgname` contribute (None => none known yet)
    :type  dependencies:   Union[dict<str, *>, None]
    :param packages:       Universe of installed Python packages within which
                           dependencies are applicable (None => all installed)
    :type  packages:       Union[dict<str, pkg_resources.Distribution>, None]
    :param package_naming: "Use package names, not project names to refer
                            to packages."
    :type  package_naming: bool

    :return: Nested dictionary (DAG) cumulating dependencies from package
             (leaf dictionaries are empty)
    :rtype:  dict<str, dict>

    .. note::
     * See `package_name()` for more about `package_naming`.
     * WARNING: DAG is constructed recursively here.
    """
    if packages is None:
        packages = get_installed_packages(package_naming=package_naming)
    dependencies = dependencies or {}
    subreqs = {package_name(p.name, package_naming): None
               for p in packages[pkgname].requires()
               if package_name(p.name, package_naming) in packages}
    dependencies[pkgname] = subreqs
    for subname in subreqs:
        build_dependencies(subname, subreqs, packages, package_naming)
    return dependencies
