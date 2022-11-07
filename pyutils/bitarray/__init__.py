# -*- mode: python -*-
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Implementation of BitArray class, providing operations and utilities
for compact (one bit per element) bit-valued arrays.
"""
# pylint:disable=too-many-lines,unnecessary-lambda
import re
from copy import copy
import operator as op


# pylint:disable=invalid-name,missing-class-docstring,missing-function-docstring,protected-access,too-few-public-methods
# noinspection PyProtectedMember
class BitArray:
    """
    Local abstraction of bitarrays: Pure-Python implementation of the canonical PyPI 'bitarray' package.

    .. note::
     * See https://pypi.python.org/pypi/bitarray for details and function documentation.
     * Provides corrections and extensions to the canonical 'bitarray' package; extensions include left/right
       shift, byte array conversions, etc.
     * The semantics of "endianness" in that module is contrary to the computing industry notion: the standard
       definition of endianness only pertains to multibyte values, and dictates the ordering of whole bytes,
       not of bits within bytes; thus, only bitarray "big" endianness is supported here, and the `endian`
       initializer keyword parameter should not be specified.
    """
    # pylint:disable=unused-argument
    # noinspection PyUnusedLocal
    def __init__(self, initval=0, endian='big'):
        if not isinstance(endian, str):
            raise TypeError
        if endian != 'big':
            raise ValueError("bitarrays have no concept of endianness")  # (bitwise "endianness" is hooey)

        for _ in (True,):
            # (Implemented as a bytearray and length.)
            if initval is None:
                initval = 0
            if isinstance(initval, bool):  # (must precede test for int)
                raise TypeError("bool initval must be an array, not a scalar")
            if isinstance(initval, int):
                self._bitlen = initval
                self._bytarr = bytearray(b'\x00' * self.size())
                break
            if isinstance(initval, self.__class__) or issubclass(self.__class__, initval.__class__):
                self._bitlen = len(initval)
                # noinspection PyUnresolvedReferences
                self._bytarr = copy(initval._bytarr)
                break
            if isinstance(initval, (bytearray, bytes)):
                self._bitlen = len(initval) * 8
                self._bytarr = bytearray(initval)
                break
            if self.isiterable(initval):
                nolen = not hasattr(initval, '__len__')
                if nolen:
                    self._bitlen = 0
                    self._bytarr = bytearray()
                else:
                    # noinspection PyTypeChecker
                    self._bitlen = len(initval)
                    self._bytarr = bytearray(b'\x00' * self.size())
                boolifier = self._bool if isinstance(initval, (str, bytes)) else bool
                for i, elem in enumerate(initval):
                    bi = i % 8
                    if bi == 0:
                        byv = 0
                    # noinspection PyUnboundLocalVariable
                    byv = byv << 1 | boolifier(elem)
                    self._bitlen += nolen
                    if bi == 7:
                        if nolen:
                            self._bytarr += b'0'
                        self._bytarr[i // 8] = byv
                rem = -(self._bitlen % -8)
                if rem:
                    if nolen:
                        self._bytarr += b'0'
                    # noinspection PyUnboundLocalVariable
                    self._bytarr[self._bitlen // 8] = byv << rem
            else:
                if not hasattr(self, '_bytarr'):
                    raise TypeError

    @staticmethod
    def isiterable(obj):
        try:
            iter(obj)
            result = True
        except TypeError:
            result = False
        return result

    def __len__(self):
        return self._bitlen

    def length(self):
        return self.__len__()

    def size(self):
        # pylint:disable=used-before-assignment
        return bits2bytes(self._bitlen)

    def _lenrem(self):
        return self._bitlen % 8

    def _lenfill(self):
        return -(self._bitlen % -8)

    @staticmethod
    def _bitmask(width):
        return (1 << width) - 1

    @staticmethod
    def _copy(dest, src):
        for m in ('_bitlen', '_bytarr'):
            setattr(dest, m, getattr(src, m))

    def copy(self):
        # noinspection PyTypeChecker
        return self.__class__(self)

    __copy__ = copy

    def _access(self):
        self._bytarr += b'0'

    def _normindex(self, _index, nocheck=False):
        _index += self._bitlen if _index < 0 else 0
        if not 0 <= _index < self._bitlen:
            if nocheck:
                _index = max(0, min(self._bitlen, _index))
            else:
                raise IndexError
        return _index

    @staticmethod
    def _bool(elem):
        if isinstance(elem, bool):
            result = elem
        elif isinstance(elem, int):
            if elem in (0, 1):
                result = bool(elem)
            else:
                raise ValueError("value must be 0 or 1, found {}".format(elem))
        elif isinstance(elem, str):
            result = False if elem == '0' else True if elem == '1' else None
            if result is None:
                raise ValueError("character must be '0' or '1', found '{}'".format(elem))
        elif isinstance(elem, bytes):
            result = False if elem == b'0' else True if elem == b'1' else None
            if result is None:
                raise ValueError("byte must be b'0' or b'1', found {}".format(elem))
        else:
            raise ValueError("expected a scalar bit value, found {}".format(elem))
        return result

    def _getitem(self, _index):
        _index = self._normindex(_index)
        return bool((self._bytarr[_index // 8] >> (7 - _index % 8)) & 1)

    def _setitem(self, _index, elem):
        byi = _index // 8
        mask = 1 << (7 - _index % 8)
        if self._bool(elem):
            self._bytarr[byi] |= mask
        else:
            self._bytarr[byi] &= ~mask

    def _getslice(self, _slice):
        return [self._getitem(i) for i in range(self._bitlen)[_slice]]

    def _setslice(self, _slice, other):
        # noinspection PyTypeChecker
        for si, di in enumerate(range(self._bitlen)[_slice]):
            elem = other[si] if hasattr(other, '__getitem__') else self._bool(other)
            self._setitem(di, elem)

    def __getitem__(self, _index):
        # noinspection PyTypeChecker
        return (self._getitem(_index) if not isinstance(_index, slice) else
                self.__class__(self._getslice(_index)))

    def __setitem__(self, _index, item):
        if not isinstance(_index, slice):
            _index = self._normindex(_index)
            self._setitem(_index, item)
        elif _index.step is not None:
            self._setslice(_index, item)
        else:
            self.__setslice__(_index.start, _index.stop, item)

    def __delitem__(self, _index):
        self._access()
        if not isinstance(_index, slice):
            _index = self._normindex(_index)
        values = self.tolist()
        del values[_index]
        # noinspection PyTypeChecker
        self.__init__(values)

    def __getslice__(self, stidx, endidx):
        # noinspection PyTypeChecker
        return self[slice(stidx, endidx)]

    def __delslice__(self, stidx, endidx):
        self.__delitem__(slice(stidx, endidx))

    def __setslice__(self, stidx, endidx, other):
        _stidx = (self._normindex(stidx, nocheck=True)
                  if stidx is not None else 0)
        _endidx = (self._normindex(endidx, nocheck=True)
                   if endidx is not None else self._bitlen)
        _isiterable = self.isiterable(other)
        if not _isiterable or _endidx - _stidx == len(other):
            for i in range(_stidx, _endidx):
                # noinspection PyTypeChecker
                self[i] = other[i - _stidx] if _isiterable else self._bool(other)
        else:
            # noinspection PyTypeChecker
            self.__init__(self._getslice(slice(0, _stidx)) +
                          self.__class__(other).tolist() +
                          self._getslice(slice(max(_stidx, _endidx), self._bitlen)))

    def _frombytes(self, bytarr, bitlen):
        rem = self._lenrem()
        offs = self.size() - 1
        self._bitlen += bitlen
        if rem == 0:
            self._bytarr += bytarr
        else:
            extn = self.size() - (offs + 1)
            if extn > 0:
                self._bytarr += bytearray(b'\x00' * extn)
            last = self._bytarr[offs]
            for byt in bytarr:
                self._bytarr[offs] = last | (byt >> rem)
                last = (byt << (8 - rem)) & 0xFF
                offs += 1
            if offs < len(self._bytarr):
                self._bytarr[offs] = last
        return self

    def _extend(self, other):
        other = self.__class__(other)
        if self._bitlen == 0:
            self._copy(self, other)
        elif other._bitlen > 0:
            self._frombytes(other._bytarr, other._bitlen)
        return self

    def extend(self, other):
        if isinstance(other, self.__class__):
            self._extend(other)
        elif not self.isiterable(other):
            raise TypeError("operand must be an aggregate, scalar found")
        else:
            try:
                newlen = self._bitlen + len(other)
                extsize = bits2bytes(newlen) - self.size()
            except (Exception, BaseException):  # (iterator has no len())
                extsize = -1

            if extsize > 0:
                self._bytarr += b'\x00' * extsize
            for elem in other:
                elem = self._bool(elem) if isinstance(other, (str, bytes)) else bool(elem)
                if extsize >= 0:
                    n = self._bitlen
                    self._bitlen += 1
                    # noinspection PyTypeChecker
                    self.__setitem__(n, elem)
                else:
                    self.append(elem)
        return self

    def append(self, elem):
        if isinstance(elem, self.__class__):
            self._extend(elem)
        else:
            n = self._bitlen
            if n % 8 == 0:
                self._bytarr += b'\x00'
            self._bitlen += 1
            # noinspection PyTypeChecker
            self.__setitem__(n, elem)
        return self

    def __add__(self, other):
        return self.copy().extend(other)

    def __iadd__(self, other):
        return self.extend(other)

    def __mul__(self, repeat):
        if repeat <= 0:
            bitarr = self.__class__()
        elif repeat == 1:
            bitarr = self
        else:
            bitarr = self.__class__(self._bitlen * repeat)
            values = self.tolist()
            for i in range(repeat):
                bitarr.__setslice__(i * self._bitlen, (i + 1) * self._bitlen, values)
        return bitarr

    __rmul__ = __mul__

    def __imul__(self, repeat):
        if repeat <= 0:
            self.__init__(0)
        elif repeat > 1:
            values = self.tolist()
            for _ in range(repeat - 1):
                self.extend(values)
        return self

    def insert(self, _index, elem):
        other = self.__class__(elem if self.isiterable(elem) else [self._bool(elem)])
        # noinspection PyTypeChecker
        self.__init__(self._getslice(slice(0, _index)) +
                      other.tolist() +
                      self._getslice(slice(_index, self._bitlen)))
        return self

    def remove(self, elem):
        self.pop(self.index(elem))
        return self

    def pop(self, _index=-1):
        _index = self._normindex(_index)
        elem = self._getitem(_index)
        self.__delslice__(_index, _index + 1)
        return elem

    def reverse(self):
        # noinspection PyTypeChecker
        self.__init__(self._getslice(slice(None, None, -1)))
        return self

    def bytereverse(self):
        for byi, byv in enumerate(self._bytarr):
            byr = 0
            for bi in range(4):
                bic = 7 - bi
                byr |= ((byv & (1 << bi)) << (bic - bi)) | ((byv & (1 << bic)) >> (bic - bi))
            self._bytarr[byi] = byr
        self._trimlast()
        return self

    def _trimlast(self):
        if self._bitlen > 0:
            self._bytarr[self.size() - 1] &= ~self._bitmask(self._lenfill())

    def setall(self, elem=True):
        elem = 0xFF if self._bool(elem) else 0x00
        for i in range(self.size()):
            self._bytarr[i] = elem
        if elem:
            self._trimlast()
        return self

    def invert(self, index=None):
        if index is not None:
            index = self._normindex(index)
            self._bytarr[index // 8] ^= 1 << (7 - index % 8)
        else:
            for i in range(self.size()):
                self._bytarr[i] ^= 0xFF
            self._trimlast()
        return self

    def _logical_op(self, logicop, other=None, insitu=False):
        if other is not None:
            if len(other) != self._bitlen:
                raise ValueError("bitarrays of equal length expected for bitwise operation")
        result = self if insitu else self.copy()
        for i in range(result.size()):
            if other is None:  # (unary)
                result._bytarr[i] = logicop(result._bytarr[i]) % 0x100
            else:              # (binary)
                result._bytarr[i] = logicop(result._bytarr[i], other._bytarr[i]) % 0x100
        result._trimlast()
        return result

    def __invert__(self):
        return self._logical_op(op.__invert__)

    def __and__(self, other):
        return self._logical_op(op.__and__, other)

    def __or__(self, other):
        return self._logical_op(op.__or__, other)

    def __xor__(self, other):
        return self._logical_op(op.__xor__, other)

    def __iand__(self, other):
        return self._logical_op(op.__and__, other, insitu=True)

    def __ior__(self, other):
        return self._logical_op(op.__or__, other, insitu=True)

    def __ixor__(self, other):
        return self._logical_op(op.__xor__, other, insitu=True)

    @staticmethod
    def _lshift(this, nbits):
        return (this if nbits == 0 else
                this.__class__(this._getslice(slice(min(nbits - len(this), 0), None)) +
                               [False] * nbits)
                if nbits > 0 else
                this.__rshift__(-nbits))

    def __lshift__(self, nbits):
        return self._lshift(self, nbits)

    @staticmethod
    def _rshift(this, nbits):
        # noinspection PyTypeChecker
        return (this if nbits == 0 else
                this.__class__([False] * nbits +
                               this._getslice(slice(None, max(len(this) - nbits, 0))))
                if nbits > 0 else
                this.__lshift__(-nbits))

    def __rshift__(self, nbits):
        return self._rshift(self, nbits)

    def __eq__(self, other):
        return self._bitlen == other._bitlen and self._bytarr == other._bytarr

    def __ne__(self, other):
        return self._bitlen != other._bitlen or self._bytarr != other._bytarr

    def __lt__(self, other):
        return self._compare_op(other, op.lt)

    def __gt__(self, other):
        return self._compare_op(other, op.gt)

    def __le__(self, other):
        return self._compare_op(other, op.le)

    def __ge__(self, other):
        return self._compare_op(other, op.ge)

    def _compare_op(self, other, cmpop):
        othmax = other.size() - 1
        for i in range(self.size()):
            if i > othmax:
                result = cmpop in (op.gt, op.ge, op.ne)
                break
            sb = self._bytarr[i]
            ob = other._bytarr[i]
            if sb != ob:
                result = cmpop(sb, ob)
                break
        else:
            result = (cmpop in (op.lt, op.le, op.ne) if self._bitlen < other._bitlen else
                      cmpop in (op.gt, op.ge, op.ne) if self._bitlen > other._bitlen else
                      cmpop in (op.le, op.ge, op.eq))
        return result

    @staticmethod
    def _find(selector, iterable, default=None, value=True, start=0, stop=None):
        result = default
        if hasattr(iterable, '__getitem__') and hasattr(iterable, '__len__'):
            for idx in range(*slice(start, stop).indices(len(iterable))):
                elem = iterable[idx]
                if selector(elem) if callable(selector) else selector[idx]:
                    result = elem if value else idx
                    break
        else:
            for idx, elem in enumerate(iterable):
                if idx < start:
                    continue
                if selector(elem) if callable(selector) else selector[idx]:
                    result = elem if value else idx
                    break
                if stop is not None and idx >= stop:
                    break
        return result

    def any(self):
        # noinspection PyTypeChecker
        return (self._bitlen > 0 and
                self._find((lambda _byt: _byt != 0x00), self._bytarr) is not None)

    def all(self):
        rem = self._lenrem()
        bytlen = len(self._bytarr) - (rem > 0)
        # noinspection PyTypeChecker
        return (self._find((lambda _byt: _byt != 0xFF), self._bytarr,
                           default=bytlen, value=False) == bytlen and
                (rem == 0 or self._bytarr[-1] >> (8 - rem) == self._bitmask(rem)))

    def index(self, elem, start=0, stop=None, default=NotImplemented):
        elem = self._bool(elem)
        start, stop, _ = slice(start, stop).indices(self._bitlen)
        startbyte = start // 8
        for _ in range(2):  # (tail end of start byte, head end of stop byte)
            idx = self._find((lambda _byt: _byt != (0x00 if elem else 0xFF)), self._bytarr,
                             default=default, value=False, start=startbyte, stop=bits2bytes(stop))
            if idx != default:
                idx *= 8
                try:
                    for idx in range(max(idx, start), min(idx + 8, stop)):
                        if self[idx] == elem:
                            break
                    else:
                        if idx + 1 < stop:
                            startbyte += 1
                            continue
                        raise ValueError
                    break
                except ValueError:
                    idx = default
                    break
        # noinspection PyUnboundLocalVariable
        if idx == NotImplemented:
            raise ValueError("bit not found")
        return idx

    def search(self, other, limit=None):
        result = []
        for i, matchpos in enumerate(self.itersearch(other)):
            if limit is not None and i >= limit:
                break
            result.append(matchpos)
        return result

    def itersearch(self, other):
        def _gen(_other):
            for match in re.finditer("(?={})".format(_other.to01()), self.to01()):
                yield match.start()

        if not isinstance(other, self.__class__):
            if not self.isiterable(other):
                other = self._bool(other)
            other = self.__class__(other)
        if hasattr(other, '__len__') and len(other) == 0:
            raise ValueError("cannot search for empty bitarray")

        return _gen(other)

    # noinspection PyTypeChecker
    def sort(self, reverse=False):
        if not isinstance(reverse, int):
            raise TypeError('reverse must be a boolean')
        count = self.count()
        order = slice(None, None, 1 if reverse else -1)
        lengths = (count, self._bitlen - count)[order]
        values = ([True], [False])[order]
        self.__init__(lengths[0] * values[0] + lengths[1] * values[1])
        return self

    def __contains__(self, elem):
        if not isinstance(elem, self.__class__):
            elem = self.__class__([self._bool(elem)] if isinstance(elem, int) else elem)
        return (hasattr(elem, '__len__') and len(elem) == 0) or len(self.search(elem, limit=1)) > 0

    def count(self, value=True, start=0, stop=None):
        count = 0
        elem = self._bool(value)
        for i in range(*slice(start, stop).indices(self._bitlen)):
            try:
                count += self[i] == elem
            except IndexError:
                break
        return count

    def __repr__(self):
        return "bitarray({1}{0}{1})".format(self.__str__(), "'"[:self._bitlen > 0])

    def __str__(self):
        return self.to01()

    def fill(self):
        nfill = self._lenfill()
        self._bitlen += nfill
        return nfill

    def clear(self):
        self._access()
        self._bytarr = bytearray()
        self._bitlen = 0
        return self

    def pack(self, bytarr):
        if not isinstance(bytarr, (bytes, bytearray)):
            raise TypeError("expecting bytes operand")
        self.extend(bytearray(bytarr))
        return self

    def unpack(self, zero=b'\x00', one=b'\xFF'):
        symbols = zero + one
        if not isinstance(symbols, bytes) or (len(zero), len(one)) != (1, 1):
            raise TypeError("symbols must be bytes")
        return bytes([symbols[elem] for elem in self])

    def frombytes(self, bytarr):
        return self._frombytes(bytarr, len(bytarr) * 8)

    def tobytes(self):
        bytarr = self.copy()._bytarr
        return bytes(bytarr)

    def tolist(self, as_ints=False):
        # noinspection PyTypeChecker
        return ([int(elem) for elem in self] if as_ints else
                self._getslice(slice(None, None)))

    def to01(self):
        # noinspection PyTypeChecker
        return ''.join(['01'[i] for i in self.tolist()])

    def fromfile(self, f, n=-1):
        chunk = f.read() if n < 0 else f.read(n)
        self.frombytes(chunk.encode() if 't' in getattr(f, 'mode', '') else chunk)
        if len(chunk) < n:
            raise EOFError
        return self

    def tofile(self, f):
        view = memoryview(self._bytarr)
        if view:
            astext = 't' in getattr(f, 'mode', '')
            if astext:
                view = self._bytarr.decode()
            f.write(view)

    def toint(self, bitlen=None, bitoffs=0, endian='big', signed=False):
        # NOTE: Here endian has meaning and is significant.
        if bitlen is None:
            bitlen = self._bitlen
        mod, first, rng = ((8, 0, range(bitlen)) if endian == 'big' else
                           (-8, bitlen + 6, range(bitlen + 6, 6, -1)))
        signbit = 0
        value = 0
        for i in rng:
            byn, bn = divmod(7 - (i + bitoffs), mod)
            bit = (self._bytarr[abs(byn)] >> abs(bn)) & 1
            if signed and i == first:
                if bit:
                    signbit = bit
            else:
                value = (value << 1) + bit
        if signed and signbit and bitlen:
            value -= 1 << (bitlen - 1)
        return value

    @classmethod
    def fromint(cls, value, other=None, bitlen=32, bitoffs=0, endian='big', signed=False):
        # NOTE: Here endian has meaning and is significant.
        mod, final, rng = ((8, 0, range(bitlen - 1, -1, -1)) if endian == 'big' else
                           (-8, bitlen + 6, range(7, bitlen + 7)))
        signbit = value < 0
        if signbit:
            twoscomp = 2 ** (bitlen - 1)
            value = -(value & twoscomp) + (value & ~twoscomp)
        bitarr = other or cls(bitlen)
        for i in rng:
            if signed and i == final:
                bit = signbit
            else:
                bit = value & 1
                value >>= 1
            byn, bn = divmod(7 - (i + bitoffs), mod)
            byn = abs(byn)
            bn = abs(bn)
            bitarr._bytarr[byn] = (bitarr._bytarr[byn] & ~(1 << bn)) | (bit << bn)
            if not other and value == 0:
                break
        return bitarr

    @staticmethod
    def endian():
        return 'big'

    def buffer_info(self):
        return (memoryview(self._bytarr),
                self.size(),
                self.endian(),
                self._lenfill() & 0x7,
                self.size())


try:
    # noinspection PyPackageRequirements
    from bitarray import bitarray
    if bitarray.__name__ != 'bitarray':
        raise ImportError
    # Corrections:
    bitarray._tobytes = bitarray.tobytes
    # noinspection PyProtectedMember
    bitarray.tobytes = \
        lambda _self: bytearray(_self._tobytes())
    bitarray._frombytes = bitarray.frombytes
    # noinspection PyProtectedMember
    bitarray.frombytes = \
        lambda _self, _bytarr: (_self._frombytes(str(_bytarr)), _self)[-1]
    # Extensions:
    # noinspection PyProtectedMember
    bitarray.__lshift__ = BitArray._lshift
    # noinspection PyProtectedMember
    bitarray.__rshift__ = BitArray._rshift
except ImportError:
    # Local substitutions:
    bitarray = BitArray

    def bitdiff(a, b):
        return (a ^ b).count()

    def bits2bytes(n):
        if isinstance(n, float):
            raise TypeError
        if n < 0:
            raise ValueError
        return (n + 7) // 8

    # pylint:disable=invalid-name,missing-function-docstring
    # noinspection PyPep8Naming
    class frozenbitarray(bitarray):
        def __repr__(self):
            return 'frozen' + bitarray.__repr__(self)

        def __hash__(self):
            if getattr(self, '_hash', None) is None:
                # pylint:disable=attribute-defined-outside-init
                self._hash = hash((len(self), self.tobytes()))
            return self._hash

        def __delitem__(self, *args, **kwargs):
            raise TypeError("'frozenbitarray' is immutable")

        append = bytereverse = clear = extend = encode = fill = __delitem__
        frombytes = fromfile = insert = invert = pack = pop = __delitem__
        remove = reverse = setall = sort = __setitem__ = __delitem__
        __iadd__ = __iand__ = __imul__ = __ior__ = __ixor__ = __delitem__


# ======================================================================================================================
# Fieldwise binary structure encoding/decoding extension:

# pylint:disable=wrong-import-position
from collections import namedtuple
import uuid
import inspect


# Endian typing variants for integer-valued fields:
# noinspection PyPep8Naming
class buint(int):
    _signed_ = False
    _endian_ = 'big'


# noinspection PyPep8Naming
class luint(int):
    _signed_ = False
    _endian_ = 'little'


# noinspection PyPep8Naming
class bsint(int):
    _signed_ = True
    _endian_ = 'big'


# noinspection PyPep8Naming
class lsint(int):
    _signed_ = True
    _endian_ = 'little'


class HashableDict:
    """ PEP rejections of a native hashable dictionary data type necessitate this miserable hack. """
    def __init__(self, **kwargs):
        # noinspection PyArgumentList
        self._tuple = namedtuple("_{}_{}".format(self.__class__.__name__,
                                                 str(uuid.uuid4()).replace('-', '')), kwargs)(**kwargs)

    def __getitem__(self, item, default=None):
        if isinstance(item, int):
            _tuple = object.__getattribute__(self, '_tuple')
            # noinspection PyProtectedMember
            item = _tuple._fields[item]
        return getattr(self, item, default)

    def __getattribute__(self, attr, default=None):
        if attr.startswith('_'):
            attrval = object.__getattribute__(self, attr)
        else:
            _tuple = object.__getattribute__(self, '_tuple')
            # noinspection PyProtectedMember
            attrval = (_tuple if attr == '_tuple' else
                       getattr(_tuple, attr, getattr(_tuple._asdict(), attr, default)))
        return attrval


class Fields(HashableDict):
    """
    Container for an ordered collection of field definitions.

    Each field is described as a (descriptor, definition) pair:
     * The `descriptor` is one of:
        - str => name for a scalar (single-valued) field (starts with '_reserved' => reserved field)
        - `CondField` => conditional descriptor, possibly involving a previously-defined field value
        - `AggField` => aggregate descriptor, possibly involving a previously-defined field value
       (either of the latter two descriptor types must be contained within a native Python dict to be expressible)
     * The definition is one of:
        - 0 => empty field (value is None)
        - 1 => boolean-valued field
        - other positive int => length (# bits) for a scalar integer-valued field
        - `bitarray` => content for a constant-valued field
        - `TypedField` => (type, length) pair denoting a data type and length (# bits) for the field
        - `Fields` or dict => a nested composite substructure field
        - `StructField` => forward reference to a named nested composite substructure field
        - ... or Ellipsis => variable-length field
    """


class StructField(Fields):
    """ Substructure Field: Forward reference to a named type containing a collection of field definitions. """
    def __init__(self, name=None, struct=None):
        super().__init__()
        self._name = name
        self._struct = struct


class CondField(HashableDict):
    """ Conditional Field: Collection of field definitions present only if specified condition is satisfied. """
    def __init__(self, fldref, oper, operand):
        super().__init__(fldref=fldref, oper=oper, operand=operand)


class AggField(HashableDict):
    """ Aggregate Field: Array of N "substructure" field definitions. """
    def __init__(self, aggtype, defref=None):
        super().__init__(aggtype=aggtype, defref=defref)


class AggLimit(HashableDict):
    """ Specifier for how to determine length of an aggregate field (by count or by offset). """
    def __init__(self, fldref, limittype=int):
        assert limittype in (bytes, int)
        super().__init__(limittype=limittype, fldref=fldref)


class TypedField(HashableDict):
    """ Specifier for how to "typecast" a field value during encoding/decoding. """
    def __init__(self, valtype, fldlen):
        super().__init__(valtype=valtype, fldlen=fldlen)


def bitarray_encode(_structdef, _instruct, check_types=True, check_reserved=True):
    """ @@@ [TBD] """
    _ = check_types, check_reserved
    raise NotImplementedError


# noinspection GrazieInspection
def bitarray_decode(_inbits, _structdef,  # noqa: C901 # pylint:disable=too-many-statements
                    check_types=True, check_reserved=True):
    """
    Decodes a binary-encoded data entity (e.g., "packet", "frame" or other blob) represented
    as a `bitarray`, using a (possibly nested) collection of structure and constant definitions
    that describe the format of the binary encoding for that entity, resulting in a (possibly
    nested) dictionary of the entity fields.

    :param _inbits:        Bit array containing the binary entity to be decoded
    :type  _inbits:        bitarray
    :param _structdef:     Structure defining the topmost format of the data entity as a
                           collection of bitwise "fields", each of which may be defined as
                           fixed, conditional, or aggregate substructures containing subfields
    :type  _structdef:     Union(dict, Fields)
    :param check_types:    "Verify that the decoded value for each field is within the domain
                           of values defined for that field."
    :type  check_types:    bool
    :param check_reserved: "Verify that fields defined as 'reserved' are zero-valued."
    :type  check_reserved: bool

    :return: Field-wise decoding of binary entity produced by applying structure format definition
    :rtype:  dict
    """
    # Implementation note: to simplify nested field lookups, a "flattened" representation of decoded fields
    # is maintained, mapping the fully-nested (dot-separated) field/subfield name to its parent structure,
    # parent structure field bit offset, field bit length, decoded field value.
    flatref = namedtuple('flatref', 'struct fldoffs fldlen fldval')

    def _ref_global(ref):
        """ Looks up a definition from the caller's namespace by name. """
        return caller_namespace.get(ref) if isinstance(ref, str) else ref

    def _ref_value(flat, prefix, ref):
        """ Looks up the (possibly nested) field value already decoded by (possibly nested) name. """
        names = ref.strip().split('.')
        if names[0]:  # (absolute reference)
            value = _ref_global(names[0])
            for fldname in names[1:]:
                if value is not None:
                    value = value.get(fldname)
        else:  # (relative reference)
            found = flat.get(ref, flat.get(prefix + ref, flat.get('.'.join(prefix.split('.')[:-1]) + ref)))
            value = found.fldval if found else None
        return value

    # pylint:disable=too-many-arguments
    def _record_field(outstruct, flddesc, flat, flddef, fullname, fldoffs, fldlen, fldval, etc):
        """
        Completes decoding of a (possibly composite) field: adds it to the most local structure in the structure
        hierarchy (and to the flattened structure representation), cumulating the field bit length onto the current
        bit offset within that local structure.

        (see `_decode_field()` for argument descriptions)
        """
        try:
            fldval = fldval['_value']  # (special case: extract aggregate field entity)
        except (Exception, BaseException):
            pass
        flat[fullname] = flatref(struct=outstruct, fldoffs=fldoffs, fldlen=fldlen, fldval=fldval)
        if etc:
            etc['defs'][flddesc] = flddef
            etc['lens'].append(fldlen)
        else:
            outstruct[flddesc] = fldval
        fldoffs += fldlen
        return fldoffs

    # pylint:disable=too-many-locals,too-many-branches,too-many-statements
    # noinspection GrazieInspection
    def _decode_field(outstruct, flat, flddesc, flddef, prefix, inbits, fldoffs, etc):
        """
        Field decoder: Decodes the binary content of a (possibly composite) field, descending down the
        (variable) hierarchy of (sub)fields and decoding each of those as necessary.

        :param outstruct: Output local (sub)structure to contain decoded field values
        :type  outstruct: dict
        :param flat:      Flattened mapping of nested field names to values, to be updated
        :type  flat:      dict
        :param flddesc:   Name of field from structure definition or descriptor for composite field composition
        :type  flddesc:   Union(str, CondField, AggField)
        :param flddef:    Field definition for field
        :type  flddef:    Union(int, dict, bitarray, Fields, TypedField, StructField, Ellipsis)
        :param prefix:    Name prefix for (sub)structure being decoded (dot-separated, '' => topmost structure)
        :type  prefix:    str
        :param inbits:    Binary data to decode; decoding starts at `fldoffs`
        :type  inbits:    bitarray
        :param fldoffs:   Bit origin within `inbits` where decoding of fields is to start
        :type  fldoffs:   int
        :param etc:       Definitions for fields following a variable-length field in parent structure (empty => none)
        :type  etc:       dict

        :return: Result:
                  [0]: Definitions for fields following a variable-length field in this structure (empty => none)
                  [1]: Length (# bits) consumed from binary data in decoding field
        :rtype:  tuple

        .. note::
         * May be called recursively, either directly or indirectly.
        """
        for _ in (True,):
            # (unconditional field)
            if isinstance(flddesc, str):
                fullname = '.'.join((prefix, flddesc))
                if isinstance(flddef, bitarray):  # (constant field)
                    fldlen = len(flddef)
                    constval = flddef.toint()
                    fldval = inbits.toint(bitlen=fldlen, bitoffs=fldoffs)
                    if fldval != constval:
                        raise ValueError(f"field '{fullname}' must have fixed value 0x{constval:X}, is 0x{fldval:X}")
                    fldoffs = _record_field(outstruct, flddesc, flat, flddef, fullname, fldoffs, fldlen, fldval, etc)
                    break

                # (variable-length field)
                if flddef == Ellipsis:
                    if etc:
                        raise TypeError("structure definition has multiple variable-length fields"
                                        f" ('{etc['fullname']}', '{fullname}')")
                    etc = dict(flddesc=flddesc, fullname=fullname, fldoffs=fldoffs, defs={}, lens=[])
                    _record_field(outstruct, flddesc, flat, flddef, fullname, fldoffs, 0, bitarray(), {})
                    break

                # (composite substructure)
                if flddef in (dict, Fields, StructField) or isinstance(flddef, (dict, Fields, StructField)):
                    structname = flddesc
                    if isinstance(flddef, StructField):
                        try:
                            flddesc = getattr(flddef, '_name') or flddesc
                        except (Exception, BaseException):
                            pass
                        try:
                            structname = getattr(flddef, '_struct') or structname
                        except (Exception, BaseException):
                            pass
                        flddef = StructField
                    if flddef in (dict, Fields, StructField):
                        flddef = caller_namespace.get(structname)
                        if not flddef:
                            raise TypeError(f"field '{fullname}' references undefined substructure {structname}")
                    substruct, fldlen = _decode_fields({}, flat, flddef, fullname, inbits, fldoffs)
                    fldoffs = _record_field(outstruct, flddesc, flat, flddef, fullname, fldoffs, fldlen, substruct, etc)
                    break

                # (scalar field)
                if isinstance(flddef, TypedField):
                    fldtype, flddef = flddef  # (typecast)
                else:
                    fldtype = int  # (simple int)
                try:
                    # noinspection PyTypeChecker
                    fldlen = int(flddef)
                except ValueError as exc:
                    if flddef == Ellipsis:  # (allowed if this is variable-length)
                        etc, fldoffs = _decode_field(outstruct, flat, flddesc, flddef, prefix, inbits, fldoffs, etc)
                        break
                    raise ValueError(f"field '{fullname}' has unknown bit length: {flddef}") from exc
                if fldlen == 0:  # (empty)
                    fldval = None
                elif fldlen == 1:  # (single-bit (bool))
                    fldval = inbits[fldoffs]
                elif fldoffs + fldlen > len(inbits):  # (truncated)
                    raise ValueError(f"structure truncated in field '{fullname}' at bit offset {fldoffs}")
                elif fldtype in (bytes, bytearray):  # (bytes-like array)
                    fldval = fldtype(inbits[fldoffs:fldoffs + fldlen].tobytes())
                else:  # (integral)
                    fldval = inbits.toint(bitlen=fldlen, bitoffs=fldoffs,
                                          endian=getattr(fldtype, '_endian_', 'big'),
                                          signed=getattr(fldtype, '_signed_', False))
                    if check_reserved and flddesc.startswith('reserved_'):
                        if fldval != 0:
                            raise ValueError(f"reserved field '{fullname}' has non-zero value 0x{fldval:X}")
                    try:
                        fldval = fldtype(fldval)
                    except ValueError as exc:
                        if check_types:
                            raise ValueError(f"field '{fullname}' is not a valid {fldtype}") from exc
                fldoffs = _record_field(outstruct, flddesc, flat, flddef, fullname, fldoffs, fldlen, fldval, etc)
                break

            # -------------
            if isinstance(flddesc, CondField):  # (conditional field)
                cond = flddesc
                try:
                    operand = cond.operand.name
                except (Exception, BaseException):
                    operand = cond.operand
                condname = '.'.join((prefix, f"CondField({cond.fldref} {cond.oper.__name__} {operand})"))
                if not callable(cond.oper):
                    raise ValueError(f"field '{condname}' triple must have a dyadic callable middle element")
                if cond.oper(_ref_value(flat, prefix, cond.fldref), cond.operand):
                    substruct, fldlen = _decode_fields(outstruct, flat, flddef, prefix, inbits, fldoffs)
                    if isinstance(flddesc, str):
                        fldoffs = _record_field(outstruct, flddesc, flat, flddef, prefix, fldoffs, fldlen, substruct,
                                                etc)
                    else:
                        fldoffs += fldlen

            # -------------
            elif isinstance(flddesc, AggField):  # (aggregation of fields)
                agg = flddesc
                aggname = '.'.join((prefix, f"AggField({agg.aggtype}({agg.defref}))"))
                rawtypes = (bytes, bytearray, bool, bitarray)
                # noinspection PyTypeChecker
                if agg.aggtype in (list, tuple) + rawtypes:
                    structdef = _ref_global(agg.defref) if agg.defref else agg.aggtype
                    if not structdef:
                        raise TypeError(f"field '{aggname}' has unknown aggregate structure reference '{agg.defref}'")
                    try:
                        limit, cntref = flddef
                    except ValueError:
                        limit, cntref = int, flddef
                    cnt = _ref_value(flat, prefix, cntref)
                    if limit in (bool, bytes):
                        cnt = fldoffs + cnt * (1, 8)[limit == bytes]
                    aggarr = []
                    agglen = 0
                    startoffs = fldoffs
                    for _ in range(1 if structdef in rawtypes else cnt):
                        if limit != int and fldoffs >= cnt:
                            break
                        if hasattr(structdef, 'items'):
                            substruct, fldlen = _decode_fields({}, flat, structdef, aggname, inbits, fldoffs)
                            aggarr.append(substruct)
                        elif structdef in rawtypes:
                            fldlen = cnt * 8
                            aggarr = inbits[fldoffs: fldoffs + fldlen]
                            if structdef == bool:
                                aggarr = aggarr.tolist()
                            elif structdef in (bytes, bytearray):
                                aggarr = aggarr.tobytes()
                        else:
                            raise TypeError(f"field '{aggname}' has unknown aggregate structure type {structdef}")
                        fldoffs += fldlen
                        agglen += fldlen
                    aggarr = agg.aggtype(aggarr)
                    fldoffs = _record_field(outstruct, '_value', flat, flddef, aggname, startoffs, agglen, aggarr, etc)
                    break
                raise TypeError(f"field '{aggname}' has unknown type definition")

        return etc, fldoffs

    # noinspection GrazieInspection
    def _decode_fields(outstruct, flat, structdef, prefix, inbits, fldoffs):
        """
        Decodes binary content for all fields within a structure, hierarchically decoding content for all subfields
        of each field that is not defined as a simple scalar value.

        :param outstruct: Output local (sub)structure to contain decoded field values
        :type  outstruct: dict
        :param flat:      Flattened mapping of nested field names to values, to be updated
        :type  flat:      dict
        :param structdef: Fieldwise definition of binary structure content
        :type  structdef: Union(dict, Fields)
        :param prefix:    Name prefix for (sub)structure being decoded (dot-separated, '' => topmost structure)
        :type  prefix:    str
        :param inbits:    Binary data to decode; decoding starts at `fldoffs`
        :type  inbits:    bitarray
        :param fldoffs:   Bit origin within `inbits` where decoding of fields is to start
        :type  fldoffs:   int

        :return: Result:
                  [0]: Decoded structure, updated with decoded field values
                  [1]: Length (# bits) consumed from binary data in decoding fields
        :rtype:  tuple

        .. note::
         * May be called recursively, either directly or indirectly.
        """
        etc = {}  # (definitions and other info for all local fields following any variable-length field definition)

        # Decode values for all fields within structure definition; values for any fields following a variable-length
        # field are not extracted/decoded -- actual offsets of those fields within the input bit blob must be reckoned
        # backward from the actual length of the blob, then extracted post facto (see immediately below).
        startoffs = fldoffs
        for flddesc, flddef in structdef.items():
            etc, fldoffs = _decode_field(outstruct, flat, flddesc, flddef, prefix, inbits, fldoffs, etc)

        # Decode values for all fields following the local variable-length field, if any.
        if etc:
            fldoffs = etc['fldoffs']
            etclen = len(inbits) - sum(etc['lens']) - fldoffs
            if etclen < 0:
                raise ValueError(f"structure truncated after variable-length field '{etc['fullname']}'"
                                 f" at bit offset {etc['fldoffs']} (short {-etclen} bits)")
            outstruct[etc['flddesc']] = flat[etc['fullname']] = inbits[fldoffs:fldoffs + etclen]
            for flddesc, flddef in etc['defs'].items():
                _, fldoffs = _decode_field(outstruct, flat, flddesc, flddef, prefix, inbits, fldoffs, {})

        return outstruct, fldoffs - startoffs

    if not isinstance(_structdef, (dict, Fields)):
        raise TypeError("first param must be dict or Fields")
    if not isinstance(_inbits, bitarray):
        raise TypeError("second param must be bitarray")

    # Extract caller's namespace wherein (sub)structure names, enums, etc. can be resolved.
    stack = inspect.stack()
    for i in range(1, len(stack)):
        if stack[i].filename != __file__:
            break
    # noinspection PyUnboundLocalVariable
    caller_namespace = [v for n, v in inspect.getmembers(stack[i][0]) if n == 'f_globals'][0]

    # Decode structure by recursive descent.
    _struct, _structlen = _decode_fields({}, {}, _structdef, '', _inbits, 0)
    assert _structlen == len(_inbits)  # (structure length must exactly match provided bitarray)
    return _struct


# Extend into bitarray implementation:
bitarray.decode = lambda self, structdef, **kwargs: bitarray_decode(self, structdef, **kwargs)
bitarray.encode = lambda structdef, instruct, **kwargs: bitarray_encode(structdef, instruct, **kwargs)
