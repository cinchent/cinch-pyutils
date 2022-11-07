"""
Tests for bitarray
"""
from __future__ import absolute_import

# pylint:disable=too-many-lines,missing-class-docstring,missing-function-docstring
# pylint:disable=invalid-name,blacklisted-name,wrong-import-position
import os
import sys
import unittest
import tempfile
import shutil
from string import hexdigits
from random import (choice, randint)

import copy
import pickle
import itertools
import binascii
from io import BytesIO

try:
    import shelve
    import hashlib
except ImportError:
    shelve = hashlib = None


def cmp(a, b):
    return (a > b) - (a < b)


try:
    # pylint:disable=import-error
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from bitarray import (bitarray, frozenbitarray, bits2bytes)
except ImportError:
    # noinspection PyUnresolvedReferences
    from pyutils.bitarray import (bitarray, frozenbitarray, bits2bytes)

tests = []


def zeros(length):
    """
    Create a bitarray of length, with all values 0.
    """
    if not isinstance(length, int):
        raise TypeError("integer expected")

    a = bitarray(length)
    a.setall(0)
    return a


# pylint:disable=too-many-return-statements
def strip(a, mode='right'):
    """
    Strip zeros from left, right or both ends.
    Allowed values for mode are the strings: `left`, `right`, `both`
    """
    if not isinstance(a, bitarray):
        raise TypeError("bitarray expected")
    if not isinstance(mode, str):
        raise TypeError("string expected for mode")
    if mode not in ('left', 'right', 'both'):
        raise ValueError("allowed values 'left', 'right', 'both', got: %r" % mode)
    first = 0
    if mode in ('left', 'both'):
        try:
            first = a.index(1)
        except ValueError:
            return bitarray(0)

    last = len(a) - 1
    if mode in ('right', 'both'):
        try:
            last = rindex(a)
        except ValueError:
            return bitarray(0)

    return a[first:last + 1]


def rindex(a, value=True):
    """
    Return the rightmost index of `bool(value)` in bitarray.
    Raises `ValueError` if the value is not present
    """
    if not hasattr(a, 'length'):
        raise TypeError
    v = bool(value)
    for i in range(a.length() - 1, -1, -1):
        if a[i] == v:
            break
    else:
        i = -1
    if i < 0:
        raise ValueError("{} not in bitarray".format(value))
    return i


def ba2hex(a):
    """
    Return a string containing with hexadecimal representation of
    the bitarray (which has to be multiple of 4 in length).
    """
    if not isinstance(a, bitarray):
        raise TypeError("bitarray expected")

    if len(a) % 4:
        raise ValueError("bitarray length not multiple of 4")

    b = a.tobytes()
    s = binascii.hexlify(b).decode()
    if len(a) % 8:
        s = s[:-1]
    return s


def hex2ba(s):
    """
    Bitarray of hexadecimal representation.
    hexstr may contain any number of hex digits (upper or lower case).
    """
    if not isinstance(s, (str, bytes)):
        raise TypeError("string expected, got: %r" % s)

    strlen = len(s)
    if strlen % 2:
        s += '0' if isinstance(s, str) else b'0'

    a = bitarray(0)
    b = binascii.unhexlify(s)
    a.frombytes(b)

    if strlen % 2:
        del a[-4:]
    return a


def ba2int(a, signed=False):
    """
    Convert the given bitarray into an integer.
    `signed` indicates whether two's complement is used to represent the integer.
    """
    if not isinstance(a, bitarray):
        raise TypeError("bitarray expected")
    length = len(a)
    if length == 0:
        raise ValueError("non-empty bitarray expected")

    if length % 8:
        a = zeros(8 - length % 8) + a
    b = a.tobytes()

    res = int.from_bytes(b, byteorder='big')

    if signed and res >= 1 << (length - 1):
        res -= 1 << length
    return res


# pylint:disable=too-many-return-statements
def int2ba(i, length=None, signed=False):
    """
    Convert the given integer to a bitarray, unless the `length` of the bitarray is provided.

    An `OverflowError` is raised if the integer is not representable with the
    given number of bits.

    `signed` determines whether two's complement is used to represent the integer,
    and requires `length` to be provided.

    If signed is False and a negative integer is given, an OverflowError is raised.
    """
    if not isinstance(i, int):
        raise TypeError("integer expected")
    if length is not None:
        if not isinstance(length, int):
            raise TypeError("integer expected for length")
        if length <= 0:
            raise ValueError("integer larger than 0 expected for length")
    if signed and length is None:
        raise TypeError("signed requires length")

    if i == 0:
        # there are special cases for 0 which we'd rather not deal with below
        return zeros(length or 1)

    if signed:
        if i >= 1 << (length - 1) or i < -(1 << (length - 1)):
            raise OverflowError("signed integer out of range")
        if i < 0:
            i += 1 << length
    elif i < 0 or (length and i >= 1 << length):
        raise OverflowError("unsigned integer out of range")

    a = bitarray(0)
    b = i.to_bytes(bits2bytes(i.bit_length()), byteorder='big')

    a.frombytes(b)
    if length is None:
        return strip(a, 'left')

    la = len(a)
    if la > length:
        # pylint:disable=invalid-unary-operand-type
        a = a[-length:]
    if la < length:
        pad = zeros(length - la)
        a = pad + a
    assert len(a) == length
    return a


def swap32(x):
    return int.from_bytes(x.to_bytes(4, byteorder='little'), byteorder='big', signed=False)


def memview(a):
    try:
        result = memoryview(a)
    except TypeError:
        # pylint:disable=protected-access
        # noinspection PyProtectedMember
        result = memoryview(a._bytarr)
    return result


# noinspection PyPep8Naming
class Util:
    @staticmethod
    def randombitarrays(start=0):
        for n in list(range(start, 25)) + [randint(1000, 2000)]:
            a = bitarray()
            a.frombytes(os.urandom(bits2bytes(n)))
            del a[n:]
            yield a

    @staticmethod
    def randomlists():
        for n in list(range(25)) + [randint(1000, 2000)]:
            yield [bool(randint(0, 1)) for _ in range(n)]

    @staticmethod
    def rndsliceidx(length):
        return None if randint(0, 1) else randint(-length - 5, length + 5)

    @staticmethod
    def slicelen(s, length):
        assert isinstance(s, slice)
        start, stop, step = s.indices(length)
        slicelength = (stop - start + (1 if step < 0 else -1)) // step + 1
        return max(slicelength, 0)

    # pylint:disable=no-member
    # noinspection PyUnresolvedReferences
    def check_obj(self, a):
        self.assertTrue(repr(type(a)).lower().rstrip('>').rstrip("'").endswith('bitarray.bitarray'))
        unused = 8 * a.buffer_info()[1] - len(a)
        self.assertTrue(0 <= unused < 8)
        self.assertEqual(unused, a.buffer_info()[3])

    # noinspection PyUnresolvedReferences
    def assertEQUAL(self, a, b):
        self.assertEqual(a, b)
        self.check_obj(a)
        self.check_obj(b)

    # pylint:disable=no-member
    # noinspection PyUnresolvedReferences
    def assertStopIteration(self, it):
        self.assertRaises(StopIteration, next, it)

    # pylint:disable=no-self-use
    # noinspection PyMethodMayBeStatic
    def assertRaisesMessage(self, excClass, msg, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except excClass as e:
            if msg != str(e):
                raise AssertionError("message: %s\n got: %s" % (msg, e)) from e


# ---------------------------------------------------------------------------
class TestsModuleFunctions(unittest.TestCase, Util):
    def test_bits2bytes(self):
        for arg in 'foo', [], None, {}, 187.0, -4.0:
            self.assertRaises(TypeError, bits2bytes, arg)

        self.assertRaises(TypeError, bits2bytes)
        self.assertRaises(TypeError, bits2bytes, 1, 2)

        self.assertRaises(ValueError, bits2bytes, -1)
        self.assertRaises(ValueError, bits2bytes, -924)

        self.assertEqual(bits2bytes(0), 0)
        for n in range(1, 100):
            m = bits2bytes(n)
            self.assertEqual(m, (n - 1) // 8 + 1)
            self.assertIsInstance(m, int)

        for n, m in [(0, 0), (1, 1), (2, 1), (7, 1), (8, 1), (9, 2),
                     (10, 2), (15, 2), (16, 2), (64, 8), (65, 9),
                     (2 ** 31, 2 ** 28), (2 ** 32, 2 ** 29), (2 ** 34, 2 ** 31),
                     (2 ** 34 + 793, 2 ** 31 + 100), (2 ** 35 - 8, 2 ** 32 - 1),
                     (2 ** 62, 2 ** 59), (2 ** 63 - 8, 2 ** 60 - 1)]:
            self.assertEqual(bits2bytes(n), m)


tests.append(TestsModuleFunctions)


# ---------------------------------------------------------------------------
class CreateObjectTests(unittest.TestCase, Util):
    def test_noInitializer(self):
        a = bitarray()
        self.assertEqual(len(a), 0)
        self.assertEqual(a.tolist(), [])
        self.check_obj(a)

    def test_integers(self):
        for n in range(50):
            a = bitarray(n)
            self.assertEqual(len(a), n)
            self.check_obj(a)

            a = bitarray(int(n))
            self.assertEqual(len(a), n)
            self.check_obj(a)

        self.assertRaises(ValueError, bitarray, -1)
        self.assertRaises(ValueError, bitarray, -924)

    def test_list(self):
        lst = ['foo', None, [1], {}]
        a = bitarray(lst)
        self.assertEqual(a.tolist(), [True, False, True, False])
        self.check_obj(a)

        for n in range(50):
            lst = [bool(randint(0, 1)) for _ in range(n)]
            a = bitarray(lst)
            self.assertEqual(a.tolist(), lst)
            self.check_obj(a)

    def test_tuple(self):
        tup = ('', True, [], {1: 2})
        a = bitarray(tup)
        self.assertEqual(a.tolist(), [False, True, False, True])
        self.check_obj(a)

        for n in range(50):
            lst = [bool(randint(0, 1)) for _ in range(n)]
            a = bitarray(tuple(lst))
            self.assertEqual(a.tolist(), lst)
            self.check_obj(a)

    def test_iter1(self):
        for n in range(50):
            lst = [bool(randint(0, 1)) for _ in range(n)]
            a = bitarray(iter(lst))
            self.assertEqual(a.tolist(), lst)
            self.check_obj(a)

    def test_iter2(self):
        for lst in self.randomlists():
            def foo():
                # pylint:disable=cell-var-from-loop
                for x in lst:
                    yield x
            a = bitarray(foo())
            self.assertEqual(a, bitarray(lst))
            self.check_obj(a)

    def test_iter3(self):
        a = bitarray(itertools.repeat(False, 10))
        self.assertEqual(a, bitarray(10 * '0'))
        # Note that the through value of '0' is True: bool('0') -> True
        a = bitarray(itertools.repeat('0', 10))
        self.assertEqual(a, bitarray(10 * '1'))

    def test_range(self):
        a = bitarray(range(-3, 3))
        self.assertEqual(a, bitarray('111011'))

    def test_01(self):
        a = bitarray('0010111')
        self.assertEqual(a.tolist(), [0, 0, 1, 0, 1, 1, 1])
        self.check_obj(a)

        for n in range(50):
            lst = [bool(randint(0, 1)) for _ in range(n)]
            s = ''.join([['0', '1'][x] for x in lst])
            a = bitarray(s)
            self.assertEqual(a.tolist(), lst)
            self.check_obj(a)

        self.assertRaises(ValueError, bitarray, '01012100')

    def test_bitarray_simple(self):
        for n in range(10):
            a = bitarray(n)
            b = bitarray(a)
            self.assertFalse(a is b)
            self.assertEQUAL(a, b)

    def test_create_empty(self):
        # pylint:disable=use-list-literal,use-dict-literal
        for x in None, 0, '', list(), tuple(), set(), dict():
            a = bitarray(x)
            self.assertEqual(len(a), 0)
            self.assertEQUAL(a, bitarray())

    def test_wrong_args(self):
        # wrong types
        for x in False, True, Ellipsis, slice(0), 0.0, 0 + 0j:
            self.assertRaises(TypeError, bitarray, x)
        # wrong values
        for x in -1, 'A':
            self.assertRaises(ValueError, bitarray, x)
        # test second (1) argument
        self.assertRaises(TypeError, bitarray, 0, None)
        self.assertRaises(TypeError, bitarray, 0, 0)
        self.assertRaises(ValueError, bitarray, 0, 'foo')
        # too many args
        self.assertRaises(TypeError, bitarray, 0, 'big', 0)


tests.append(CreateObjectTests)


# ---------------------------------------------------------------------------
class TestsZeros(unittest.TestCase):
    def test_1(self):
        a = zeros(0)
        self.assertEqual(a, bitarray())
        for n in range(100):
            a = zeros(n)
            self.assertEqual(a, bitarray(n * '0'))

    def test_wrong_args(self):
        self.assertRaises(TypeError, zeros)  # no argument
        self.assertRaises(TypeError, zeros, '')
        self.assertRaises(TypeError, zeros, bitarray())
        self.assertRaises(TypeError, zeros, [])
        self.assertRaises(TypeError, zeros, 1.0)
        self.assertRaises(ValueError, zeros, -1)


tests.append(TestsZeros)


# ---------------------------------------------------------------------------
class TestsRindex(unittest.TestCase, Util):
    # noinspection PyTypeChecker
    def test_simple(self):
        self.assertRaises(TypeError, rindex)
        self.assertRaises(TypeError, rindex, None)

        a = bitarray('00010110000')
        self.assertEqual(rindex(a), 6)
        self.assertEqual(rindex(a, 1), 6)
        self.assertEqual(rindex(a, 'A'), 6)
        self.assertEqual(rindex(a, True), 6)

        a = bitarray('00010110111')
        self.assertEqual(rindex(a, 0), 7)
        self.assertEqual(rindex(a, None), 7)
        self.assertEqual(rindex(a, False), 7)

        a = frozenbitarray('00010110111')
        self.assertEqual(rindex(a, 0), 7)
        self.assertEqual(rindex(a, None), 7)
        self.assertEqual(rindex(a, False), 7)

        for v in 0, 1:
            self.assertRaises(ValueError, rindex, bitarray(0), v)
        self.assertRaises(ValueError, rindex, bitarray('000'), 1)
        self.assertRaises(ValueError, rindex, bitarray('11111'), 0)

    def test_random(self):
        for a in self.randombitarrays():
            v = randint(0, 1)
            try:
                # noinspection PyTypeChecker
                i = rindex(a, v)
            except ValueError:
                i = None
            s = a.to01()
            try:
                j = s.rindex(str(v))
            except ValueError:
                j = None
            self.assertEqual(i, j)

    def test_3(self):
        for _ in range(10):
            n = randint(1, 100000)
            v = randint(0, 1)
            a = bitarray(n)
            a.setall(1 - v)
            lst = [randint(0, n - 1) for _ in range(100)]
            for i in lst:
                a[i] = v
            # noinspection PyTypeChecker
            self.assertEqual(rindex(a, v), max(lst))

    def test_one_set(self):
        for _ in range(10):
            n = randint(1, 10000)
            a = bitarray(n)
            a.setall(0)
            a[randint(0, n - 1)] = 1
            self.assertEqual(rindex(a), a.index(1))


tests.append(TestsRindex)


# ---------------------------------------------------------------------------
class TestsStrip(unittest.TestCase, Util):
    def test_simple(self):
        self.assertRaises(TypeError, strip, '0110')
        self.assertRaises(TypeError, strip, bitarray(), 123)
        self.assertRaises(ValueError, strip, bitarray(), 'up')

        a = bitarray('00010110000')
        self.assertEQUAL(strip(a), bitarray('0001011'))
        self.assertEQUAL(strip(a, 'left'), bitarray('10110000'))
        self.assertEQUAL(strip(a, 'both'), bitarray('1011'))
        b = frozenbitarray('00010110000')
        self.assertEqual(strip(b, 'both'), bitarray('1011'))

        for mode in 'left', 'right', 'both':
            self.assertEqual(strip(bitarray('000'), mode), bitarray())
            self.assertEqual(strip(bitarray(), mode), bitarray())

    def test_random(self):
        for a in self.randombitarrays():
            b = a.copy()
            s = a.to01()
            self.assertEqual(strip(a, 'left'), bitarray(s.lstrip('0')))
            self.assertEqual(strip(a, 'right'), bitarray(s.rstrip('0')))
            self.assertEqual(strip(a, 'both'), bitarray(s.strip('0')))
            self.assertEQUAL(a, b)

    def test_one_set(self):
        for _ in range(100):
            n = randint(1, 10000)
            a = bitarray(n)
            a.setall(0)
            a[randint(0, n - 1)] = 1
            self.assertEqual(strip(a, 'both'), bitarray('1'))


tests.append(TestsStrip)


# ---------------------------------------------------------------------------
CODEDICT = {
    '0': bitarray('0000'),    '1': bitarray('0001'),
    '2': bitarray('0010'),    '3': bitarray('0011'),
    '4': bitarray('0100'),    '5': bitarray('0101'),
    '6': bitarray('0110'),    '7': bitarray('0111'),
    '8': bitarray('1000'),    '9': bitarray('1001'),
    'a': bitarray('1010'),    'b': bitarray('1011'),
    'c': bitarray('1100'),    'd': bitarray('1101'),
    'e': bitarray('1110'),    'f': bitarray('1111'),
}


# ---------------------------------------------------------------------------
class TestsHexlify(unittest.TestCase, Util):
    def test_ba2hex(self):
        self.assertEqual(ba2hex(bitarray(0)), '')
        self.assertEqual(ba2hex(bitarray('1110')), 'e')
        self.assertEqual(ba2hex(bitarray('00000001')), '01')
        self.assertEqual(ba2hex(bitarray('10000000')), '80')
        self.assertEqual(ba2hex(frozenbitarray('11000111')), 'c7')
        # length not multiple of 4
        self.assertRaises(ValueError, ba2hex, bitarray('10'))
        self.assertRaises(TypeError, ba2hex, '101')

        c = ba2hex(bitarray('1101'))
        self.assertIsInstance(c, str)

        for n in range(7):
            a = bitarray(n * '1111')
            b = a.copy()
            self.assertEqual(ba2hex(a), n * 'f')
            # ensure original object wasn't altered
            self.assertEQUAL(a, b)

    def test_hex2ba(self):
        self.assertEqual(hex2ba(''), bitarray())
        for c in 'e', 'E', b'e', b'E':
            a = hex2ba(c)
            self.assertEqual(a.to01(), '1110')
        self.assertEQUAL(hex2ba('01'), bitarray('00000001'))
        self.assertRaises(Exception, hex2ba, '01a7x89')
        self.assertRaises(TypeError, hex2ba, 0)

    @staticmethod
    def hex2ba(s):
        a = bitarray(0)
        for d in s:
            a.extend(CODEDICT[d])
        return a

    @staticmethod
    def ba2hex(a):
        s = ''
        k = list(CODEDICT.keys())
        v = list(CODEDICT.values())
        for i in range(0, len(a), 4):
            d = v.index(bitarray(a[i: i + 4]))
            s += k[d]
        return s

    def test_explicit(self):
        data = [
            ('',     ''),
            ('0000', '0'),  ('0001', '1'),
            ('1000', '8'),  ('1001', '9'),
            ('0100', '4'),  ('0101', '5'),
            ('1100', 'c'),  ('1101', 'd'),
            ('0010', '2'),  ('0011', '3'),
            ('1010', 'a'),  ('1011', 'b'),
            ('0110', '6'),  ('0111', '7'),
            ('1110', 'e'),  ('1111', 'f'),
            ('10001100',             '8c'),
            ('100011001110',         '8ce'),
            ('1000110011101111',     '8cef'),
            ('10001100111011110100', '8cef4'),
        ]
        for bs, hex_be in data:
            a_be = bitarray(bs)
            self.assertEQUAL(hex2ba(hex_be), a_be)
            self.assertEqual(ba2hex(a_be), hex_be)
            self.assertEQUAL(self.hex2ba(hex_be), a_be)
            self.assertEqual(self.ba2hex(a_be), hex_be)

    def test_round_trip(self):
        for _ in range(100):
            s = ''.join(choice(hexdigits) for _ in range(randint(0, 1000)))
            a = hex2ba(s)
            self.assertEqual(len(a) % 4, 0)
            t = ba2hex(a)
            self.assertEqual(t, s.lower())
            b = hex2ba(t)
            self.assertEQUAL(a, b)
            self.assertEQUAL(a, self.hex2ba(t))
            self.assertEqual(t, self.ba2hex(a))


tests.append(TestsHexlify)


# ---------------------------------------------------------------------------
class TestsIntegerization(unittest.TestCase, Util):
    def test_ba2int(self):
        self.assertEqual(ba2int(bitarray('0')), 0)
        self.assertEqual(ba2int(bitarray('1')), 1)
        self.assertEqual(ba2int(bitarray('00101')), 5)
        self.assertEqual(ba2int(frozenbitarray('11')), 3)
        self.assertRaises(ValueError, ba2int, bitarray())
        self.assertRaises(ValueError, ba2int, frozenbitarray())
        self.assertRaises(TypeError, ba2int, '101')
        a = bitarray('111')
        b = a.copy()
        self.assertEqual(ba2int(a), 7)
        # ensure original object wasn't altered
        self.assertEQUAL(a, b)

    def test_int2ba(self):
        self.assertEqual(int2ba(0), bitarray('0'))
        self.assertEqual(int2ba(1), bitarray('1'))
        self.assertEqual(int2ba(5), bitarray('101'))
        self.assertEQUAL(int2ba(6), bitarray('110'))
        self.assertRaises(TypeError, int2ba, 1.0)
        self.assertRaises(TypeError, int2ba, 1, 3.0)
        self.assertRaises(ValueError, int2ba, 1, 0)
        # signed integer requires length
        self.assertRaises(TypeError, int2ba, 100, signed=True)

    def test_unsigned_int(self):
        unsigned_tests = [
            ('00000000000000000000000000000000', '00000000000000000000000000000000', 0),
            ('00000000000000000000000000000001', '00000001000000000000000000000000', 1),
            ('00000000000000000000000000000010', '00000010000000000000000000000000', 2),
            ('00000000000000000000000000000111', '00000111000000000000000000000000', 7),
            ('00000000000000000000000000011110', '00011110000000000000000000000000', 30),
            ('00000000000000000000000111111110', '11111110000000010000000000000000', 510),
            ('01110110000010100101000000001001', '00001001010100000000101001110110', 0x760A5009),
            ('00000000000111111111111111111111', '11111111111111110001111100000000', 2 ** 21 - 1),
            ('11111111111111111111111111111111', '11111111111111111111111111111111', 2 ** 32 - 1),
        ]

        for bab, bal, i in unsigned_tests:
            self.assertEqual(bitarray(bab).toint(), i)
            self.assertEqual(bitarray(bal).toint(endian='little'), i)

        for bab, bal, i in unsigned_tests:
            self.assertEqual(bitarray(bab), bitarray.fromint(i))
            self.assertEqual(bitarray(bal), bitarray.fromint(i, endian='little'))

    def test_signed_int(self):
        signed_tests = [
            ('00000000000000000000000000000000', '00000000000000000000000000000000', 0),
            ('00000000000000000000000000000001', '00000001000000000000000000000000', 1),
            ('11111111111111111111111111111111', '11111111111111111111111111111111', -1),
            ('00000000000000000000000000000010', '00000010000000000000000000000000', 2),
            ('11111111111111111111111111111110', '11111110111111111111111111111111', -2),
            ('01111110010110101000011001000010', '01000010100001100101101001111110', 0x7E5A8642),
            ('10000001101001010111100110111110', '10111110011110011010010110000001', -0x7E5A8642),
            ('00000000000111111111111111111111', '11111111111111110001111100000000', 2 ** 21 - 1),
            ('11111111111000000000000000000001', '00000001000000001110000011111111', 1 - 2 ** 21),
            ('01111111111111111111111111111111', '11111111111111111111111101111111', 2 ** 31 - 1),
            ('10000000000000000000000000000001', '00000001000000000000000010000000', 1 - 2 ** 31),
        ]

        for bab, bal, i in signed_tests:
            self.assertEqual(bitarray(bab).toint(signed=True), i)
            self.assertEqual(bitarray(bal).toint(signed=True, endian='little'), i)

        for bab, bal, i in signed_tests:
            self.assertEqual(bitarray(bab), bitarray.fromint(i, signed=True))
            self.assertEqual(bitarray(bal), bitarray.fromint(i, signed=True, endian='little'))

    def test_int2ba_overflow(self):
        self.assertRaises(OverflowError, int2ba, -1)
        self.assertRaises(OverflowError, int2ba, -1, 4)

        self.assertRaises(OverflowError, int2ba, 128, 7)
        self.assertRaises(OverflowError, int2ba, 64, 7, signed=1)
        self.assertRaises(OverflowError, int2ba, -65, 7, signed=1)

        for n in range(1, 20):
            self.assertRaises(OverflowError, int2ba, 2 ** n, n)
            self.assertRaises(OverflowError, int2ba, 2 ** (n - 1), n, signed=1)
            self.assertRaises(OverflowError, int2ba, -2 ** (n - 1) - 1, n, signed=1)

    def test_int2ba_length(self):
        self.assertRaises(TypeError, int2ba, 0, 1.0)
        self.assertRaises(ValueError, int2ba, 0, 0)
        self.assertEqual(int2ba(5, length=6), bitarray('000101'))
        for n in range(1, 100):
            ab = int2ba(1, n)
            self.assertEqual(len(ab), n)
            self.assertEqual(ab, bitarray((n - 1) * '0') + bitarray('1'))

            ab = int2ba(0, n)
            self.assertEqual(len(ab), n)
            self.assertEqual(ab, bitarray(n * '0'))

            self.assertEqual(int2ba(2 ** n - 1), bitarray(n * '1'))
            self.assertEqual(int2ba(-1, n, signed=True), bitarray(n * '1'))

    def test_explicit(self):
        for i, sa in [(0,      '0'),    (1,         '1'),
                      (2,     '10'),    (3,        '11'),
                      (25, '11001'),  (265, '100001001'),
                      (3691038, '1110000101001000011110')]:
            ab = bitarray(sa)
            self.assertEQUAL(int2ba(i), ab)

    def check_round_trip(self, i):
        a = int2ba(i)
        self.assertTrue(len(a) > 0)
        # ensure we have no leading zeros
        self.assertTrue(len(a) == 1 or a.index(1) == 0)
        self.assertEqual(ba2int(a), i)
        if i > 0:
            self.assertEqual(i.bit_length(), len(a))
        # add a few trailing / leading zeros to bitarray
        a = zeros(randint(0, 3)) + a
        self.assertEqual(ba2int(a), i)

    def test_many(self):
        for i in range(100):
            self.check_round_trip(i)
            self.check_round_trip(randint(0, 10 ** randint(3, 300)))

    @staticmethod
    def twos_complement(i, num_bits):
        # https://en.wikipedia.org/wiki/Two%27s_complement
        mask = 2 ** (num_bits - 1)
        return -(i & mask) + (i & ~mask)

    def test_random_signed(self):
        for a in self.randombitarrays(start=1):
            i = ba2int(a, signed=True)
            b = int2ba(i, len(a), signed=True)
            self.assertEQUAL(a, b)

            j = ba2int(a, signed=False)  # unsigned
            if i >= 0:
                self.assertEqual(i, j)

            self.assertEqual(i, self.twos_complement(j, len(a)))


tests.append(TestsIntegerization)


# ---------------------------------------------------------------------------
class ToObjectsTests(unittest.TestCase, Util):
    def test_numeric(self):
        a = bitarray()
        self.assertRaises(Exception, int, a)
        self.assertRaises(Exception, float, a)
        self.assertRaises(Exception, complex, a)

    def test_list(self):
        for a in self.randombitarrays():
            self.assertEqual(list(a), a.tolist())

    def test_tuple(self):
        for a in self.randombitarrays():
            self.assertEqual(tuple(a), tuple(a.tolist()))


tests.append(ToObjectsTests)


# ---------------------------------------------------------------------------
class MetaDataTests(unittest.TestCase, Util):
    def test_buffer_info1(self):
        a = bitarray(13)
        self.assertEqual(a.buffer_info()[1:4], (2, 'big', 3))

        a = bitarray()
        self.assertRaises(TypeError, a.buffer_info, 42)

        bi = a.buffer_info()
        self.assertIsInstance(bi, tuple)
        self.assertEqual(len(bi), 5)
        self.assertEqual(bi[2], 'big')

    def test_buffer_info2(self):
        for n in range(50):
            bi = bitarray(n).buffer_info()
            self.assertEqual(bi[1], bits2bytes(n))  # bytes
            self.assertEqual(bi[2], 'big')          # endianness
            self.assertEqual(bi[3], 8 * bi[1] - n)  # unused
            self.assertTrue(bi[4] >= bi[1])         # allocated

    def test_len(self):
        for n in range(100):
            a = bitarray(n)
            self.assertEqual(len(a), n)


tests.append(MetaDataTests)


# ---------------------------------------------------------------------------
class SliceTests(unittest.TestCase, Util):
    def test_getitem1(self):
        a = bitarray()
        self.assertRaises(IndexError, a.__getitem__,  0)
        a.append(True)
        self.assertEqual(a[0], True)
        self.assertRaises(IndexError, a.__getitem__,  1)
        self.assertRaises(IndexError, a.__getitem__, -2)
        a.append(False)
        self.assertEqual(a[1], False)
        self.assertRaises(IndexError, a.__getitem__,  2)
        self.assertRaises(IndexError, a.__getitem__, -3)

    def test_getitem2(self):
        a = bitarray('1100010')
        for i, b in enumerate([True, True, False, False, False, True, False]):
            self.assertEqual(a[i], b)
            self.assertEqual(a[i - 7], b)
        self.assertRaises(IndexError, a.__getitem__,  7)
        self.assertRaises(IndexError, a.__getitem__, -8)

    def test_getitem3(self):
        a = bitarray('0100000100001')
        self.assertEQUAL(a[:], a)
        self.assertFalse(a[:] is a)
        aa = a.tolist()
        self.assertEQUAL(a[11:2:-3], bitarray(aa[11:2:-3]))
        self.check_obj(a[:])

        self.assertRaises(ValueError, a.__getitem__, slice(None, None, 0))
        self.assertRaises(TypeError, a.__getitem__, (1, 2))

    def test_getitem4(self):
        for a in self.randombitarrays(start=1):
            aa = a.tolist()
            la = len(a)
            for _ in range(10):
                step = self.rndsliceidx(la) or None
                s = slice(self.rndsliceidx(la), self.rndsliceidx(la), step)
                self.assertEQUAL(a[s], bitarray(aa[s]))

    def test_setitem1(self):
        a = bitarray([False])
        a[0] = 1
        self.assertEqual(a, bitarray('1'))

        a = bitarray(2)
        a[0] = 0
        a[1] = 1
        self.assertEqual(a, bitarray('01'))
        a[-1] = 0
        a[-2] = 1
        self.assertEqual(a, bitarray('10'))

        self.assertRaises(IndexError, a.__setitem__,  2, True)
        self.assertRaises(IndexError, a.__setitem__, -3, False)

    def test_setitem2(self):
        for a in self.randombitarrays(start=1):
            la = len(a)
            i = randint(0, la - 1)
            aa = a.tolist()
            ida = id(a)
            val = bool(randint(0, 1))
            a[i] = val
            aa[i] = val
            self.assertEqual(a.tolist(), aa)
            self.assertEqual(id(a), ida)
            self.check_obj(a)

            b = bitarray(la)
            b[0:la] = bitarray(a)
            self.assertEqual(a, b)
            self.assertNotEqual(id(a), id(b))

            b = bitarray(la)
            b[:] = bitarray(a)
            self.assertEqual(a, b)
            self.assertNotEqual(id(a), id(b))

            b = bitarray(la)
            b[::-1] = bitarray(a)
            self.assertEqual(a.tolist()[::-1], b.tolist())

    def test_setitem3(self):
        a = bitarray('00000')
        a[0] = 1
        a[-2] = 1
        self.assertEqual(a, bitarray('10010'))
        self.assertRaises(IndexError, a.__setitem__, 5, 'foo')
        self.assertRaises(IndexError, a.__setitem__, -6, 'bar')

    def test_setitem4(self):
        for a in self.randombitarrays(start=1):
            la = len(a)
            for _ in range(10):
                step = self.rndsliceidx(la) or None
                s = slice(self.rndsliceidx(la), self.rndsliceidx(la), step)
                lb = randint(0, 10) if step is None else self.slicelen(s, la)
                b = bitarray(lb)
                c = bitarray(a)
                c[s] = b
                self.check_obj(c)
                cc = a.tolist()
                cc[s] = b.tolist()
                self.assertEqual(c, bitarray(cc))

    def test_setslice_to_bitarray(self):
        a = bitarray('11111111' '1111')
        a[2:6] = bitarray('0010')
        self.assertEqual(a, bitarray('11001011' '1111'))
        a.setall(0)
        a[::2] = bitarray('111001')
        self.assertEqual(a, bitarray('10101000' '0010'))
        a.setall(0)
        a[3:] = bitarray('111')
        self.assertEqual(a, bitarray('000111'))
        a = bitarray(12)
        a.setall(0)
        a[1:11:2] = bitarray('11101')
        self.assertEqual(a, bitarray('01010100' '0100'))
        a = bitarray(12)
        a.setall(0)
        a[:-6:-1] = bitarray('10111')
        self.assertEqual(a, bitarray('00000001' '1101'))
        a = bitarray('1111')
        a[3:3] = bitarray('000')  # insert
        self.assertEqual(a, bitarray('1110001'))
        a[2:5] = bitarray()  # remove
        self.assertEqual(a, bitarray('1101'))
        a = bitarray('1111')
        a[1:3] = bitarray('0000')
        self.assertEqual(a, bitarray('100001'))
        a[:] = bitarray('010')  # replace all values
        self.assertEqual(a, bitarray('010'))

    def test_setslice_to_bool(self):
        a = bitarray('' + '11111111')
        a[::2] = False  # _^ ^ ^ ^
        self.assertEqual(a, bitarray('01010101'))
        a[4::] = True  # _________________^^^^
        self.assertEqual(a, bitarray('01011111'))
        a[-2:] = False  # __________________^^
        self.assertEqual(a, bitarray('01011100'))
        a[:2:] = True  # _____________^^
        self.assertEqual(a, bitarray('11011100'))
        a[:] = True  # _______________^^^^^^^^
        self.assertEqual(a, bitarray('11111111'))
        a[2:5] = False  # ______________^^^
        self.assertEqual(a, bitarray('11000111'))
        a[1::3] = False  # ____________^  ^  ^
        self.assertEqual(a, bitarray('10000110'))
        a[1:6:2] = True  # ____________^ ^ ^
        self.assertEqual(a, bitarray('11010110'))

    def test_setslice_to_int(self):
        a = bitarray('11111111')
        a[::2] = 0  # ^ ^ ^ ^
        self.assertEqual(a, bitarray('01010101'))
        a[4::] = 1  # ____________________^^^^
        self.assertEqual(a, bitarray('01011111'))
        a.__setitem__(slice(-2, None, None), 0)
        self.assertEqual(a, bitarray('01011100'))
        self.assertRaises(ValueError, a.__setitem__, slice(None, None, 2), 3)
        self.assertRaises(ValueError, a.__setitem__, slice(None, 2, None), -1)

    def test_sieve(self):  # Sieve of Eratosthenes
        a = bitarray(50)
        a.setall(1)
        for i in range(2, 8):
            if a[i]:
                a[i * i::i] = 0
        primes = [i for i in range(2, 50) if a[i]]
        self.assertEqual(primes, [2, 3, 5, 7, 11, 13, 17, 19,
                                  23, 29, 31, 37, 41, 43, 47])

    def test_delitem1(self):
        a = bitarray('100110')
        del a[1]
        self.assertEqual(len(a), 5)
        del a[3]
        del a[-2]
        self.assertEqual(a, bitarray('100'))
        self.assertRaises(IndexError, a.__delitem__,  3)
        self.assertRaises(IndexError, a.__delitem__, -4)
        a = bitarray('10101100' '10110')
        del a[3:9]  # ___^^^^^   ^
        self.assertEqual(a, bitarray('1010110'))
        del a[::3]  # ________________^  ^  ^
        self.assertEqual(a, bitarray('0111'))
        a = bitarray('10101100' '101101111')
        del a[5:-3:3]  # __^     ^  ^
        self.assertEqual(a, bitarray('1010100' '0101111'))
        a = bitarray('10101100' '1011011')
        del a[:-9:-2]  # ________^ ^ ^ ^
        self.assertEqual(a, bitarray('10101100' '011'))

    def test_delitem2(self):
        for a in self.randombitarrays(start=1):
            la = len(a)
            for _ in range(10):
                step = self.rndsliceidx(la) or None
                s = slice(self.rndsliceidx(la), self.rndsliceidx(la), step)
                c = a.copy()
                del c[s]
                self.check_obj(c)
                c_lst = a.tolist()
                del c_lst[s]
                self.assertEQUAL(c, bitarray(c_lst))


tests.append(SliceTests)


# ---------------------------------------------------------------------------
class MiscTests(unittest.TestCase, Util):
    def test_instancecheck(self):
        a = bitarray('011')
        self.assertIsInstance(a, bitarray)
        self.assertFalse(isinstance(a, str))

    def test_booleanness(self):
        self.assertEqual(bool(bitarray('')), False)
        self.assertEqual(bool(bitarray('0')), True)
        self.assertEqual(bool(bitarray('1')), True)

    def test_to01(self):
        a = bitarray()
        self.assertEqual(a.to01(), '')
        self.assertIsInstance(a.to01(), str)

        a = bitarray('101')
        self.assertEqual(a.to01(), '101')
        self.assertIsInstance(a.to01(), str)

    def test_iterate(self):
        for lst in self.randomlists():
            acc = []
            for b in bitarray(lst):
                acc.append(b)
            self.assertEqual(acc, lst)

    def test_iter1(self):
        it = iter(bitarray('011'))
        self.assertEqual(next(it), False)
        self.assertEqual(next(it), True)
        self.assertEqual(next(it), True)
        self.assertStopIteration(it)

    def test_iter2(self):
        for a in self.randombitarrays():
            aa = a.tolist()
            self.assertEqual(list(a), aa)
            self.assertEqual(list(iter(a)), aa)

    def test_assignment(self):
        a = bitarray('00110111001')
        a[1:3] = a[7:9]
        a[-1:] = a[:1]
        b = bitarray('01010111000')
        self.assertEqual(a, b)

    def test_compare_eq_ne(self):
        for n in range(1, 20):
            a = bitarray(n)
            a.setall(1)
            b = bitarray(n)
            b.setall(1)
            self.assertTrue(a == b)
            self.assertFalse(a != b)
            b[n - 1] = not b[n - 1]  # flip last bit
            self.assertTrue(a != b)
            self.assertFalse(a == b)

    def test_compare_random(self):
        for a in self.randombitarrays():
            aa = a.tolist()
            for b in self.randombitarrays():
                bb = b.tolist()
                self.assertEqual(a == b, aa == bb)
                self.assertEqual(a != b, aa != bb)
                self.assertEqual(a <= b, aa <= bb)
                self.assertEqual(a < b, aa < bb)
                self.assertEqual(a >= b, aa >= bb)
                self.assertEqual(a > b, aa > bb)

    def test_subclassing(self):
        # pylint:disable=too-few-public-methods
        class ExaggeratingBitarray(bitarray):
            def __init__(self, _data, offset):
                super().__init__(_data)
                self.offset = offset

            def __getitem__(self, it):
                return bitarray.__getitem__(self, it - self.offset)

        for a in self.randombitarrays(start=0):
            b = ExaggeratingBitarray(a, 1234)
            # pylint:disable=consider-using-enumerate
            for i in range(len(a)):
                self.assertEqual(a[i], b[i + 1234])

    def test_pickle(self):
        for a in self.randombitarrays():
            b = pickle.loads(pickle.dumps(a))
            self.assertFalse(b is a)
            self.assertEQUAL(a, b)

    def test_str_create(self):
        a = bitarray(str())
        self.assertEqual(a, bitarray())

        a = bitarray(str('111001'))
        self.assertEqual(a, bitarray('111001'))

        for a in self.randombitarrays():
            b = bitarray(str(a.to01()))
            self.assertEqual(a, b)

    def test_str_extend(self):
        a = bitarray()
        a.extend(str())
        self.assertEqual(a, bitarray())

        a = bitarray()
        a.extend(str('001011'))
        self.assertEqual(a, bitarray('001011'))

        for a in self.randombitarrays():
            b = bitarray()
            b.extend(str(a.to01()))
            self.assertEqual(a, b)

    def test_unhashable(self):
        a = bitarray()
        self.assertRaises(TypeError, hash, a)
        self.assertRaises(TypeError, dict, [(a, 'foo')])


tests.append(MiscTests)


# ---------------------------------------------------------------------------
class SpecialMethodTests(unittest.TestCase, Util):
    def test_all(self):
        a = bitarray()
        self.assertTrue(a.all())
        for s, r in ('0', False), ('1', True), ('01', False):
            self.assertEqual(bitarray(s).all(), r)

        for a in self.randombitarrays():
            self.assertEqual(all(a), a.all())
            self.assertEqual(all(a.tolist()), a.all())

    def test_any(self):
        a = bitarray()
        self.assertFalse(a.any())
        for s, r in ('0', False), ('1', True), ('01', True):
            self.assertEqual(bitarray(s).any(), r)

        for a in self.randombitarrays():
            self.assertEqual(any(a), a.any())
            self.assertEqual(any(a.tolist()), a.any())

    def test_repr(self):
        r = repr(bitarray())
        self.assertEqual(r, "bitarray()")
        self.assertIsInstance(r, str)

        r = repr(bitarray('10111'))
        self.assertEqual(r, "bitarray('10111')")
        self.assertIsInstance(r, str)

        for a in self.randombitarrays():
            b = eval(repr(a))  # pylint:disable=eval-used
            self.assertFalse(b is a)
            self.assertEqual(a, b)
            self.check_obj(b)

    def test_copy(self):
        for a in self.randombitarrays():
            b = a.copy()
            self.assertFalse(b is a)
            self.assertEQUAL(a, b)

            b = copy.copy(a)
            self.assertFalse(b is a)
            self.assertEQUAL(a, b)

            b = copy.deepcopy(a)
            self.assertFalse(b is a)
            self.assertEQUAL(a, b)

    def assertReallyEqual(self, a, b):
        # assertEqual first, because it will have a good message if the
        # assertion fails.
        self.assertEqual(a, b)
        self.assertEqual(b, a)
        self.assertTrue(a == b)
        self.assertTrue(b == a)
        self.assertFalse(a != b)
        self.assertFalse(b != a)

    def assertReallyNotEqual(self, a, b):
        # assertNotEqual first, because it will have a good message if the
        # assertion fails.
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, a)
        self.assertFalse(a == b)
        self.assertFalse(b == a)
        self.assertTrue(a != b)
        self.assertTrue(b != a)

    def test_equality(self):
        self.assertReallyEqual(bitarray(''), bitarray(''))
        self.assertReallyEqual(bitarray('0'), bitarray('0'))
        self.assertReallyEqual(bitarray('1'), bitarray('1'))

    def test_not_equality(self):
        self.assertReallyNotEqual(bitarray(''), bitarray('1'))
        self.assertReallyNotEqual(bitarray(''), bitarray('0'))
        self.assertReallyNotEqual(bitarray('0'), bitarray('1'))

    def test_equality_random(self):
        for a in self.randombitarrays(start=1):
            b = a.copy()
            self.assertReallyEqual(a, b)
            n = len(a)
            b.invert(n - 1)  # flip last bit
            self.assertReallyNotEqual(a, b)


tests.append(SpecialMethodTests)


# ---------------------------------------------------------------------------
class SequenceMethodsTests(unittest.TestCase, Util):
    def test_concat(self):
        c = bitarray('001') + bitarray('110')
        self.assertEQUAL(c, bitarray('001110'))

        for a in self.randombitarrays():
            aa = a.copy()
            for b in self.randombitarrays():
                bb = b.copy()
                c = a + b
                self.assertEqual(c, bitarray(a.tolist() + b.tolist()))
                self.check_obj(c)

                self.assertEQUAL(a, aa)
                self.assertEQUAL(b, bb)

        a = bitarray()
        self.assertRaises(TypeError, a.__add__, 42)

    def test_inplace_concat(self):
        c = bitarray('001')
        c += bitarray('110')
        self.assertEqual(c, bitarray('001110'))
        c += '111'
        self.assertEqual(c, bitarray('001110111'))

        for a in self.randombitarrays():
            for b in self.randombitarrays():
                c = bitarray(a)
                d = c
                d += b
                self.assertEqual(d, a + b)
                self.assertTrue(c is d)
                self.assertEQUAL(c, d)
                self.check_obj(d)

        a = bitarray()
        self.assertRaises(TypeError, a.__iadd__, 42)

    def test_repeat(self):
        for c in [0 * bitarray(),
                  0 * bitarray('1001111'),
                  -1 * bitarray('100110'),
                  11 * bitarray()]:
            self.assertEQUAL(c, bitarray())

        c = 3 * bitarray('001')
        self.assertEQUAL(c, bitarray('001001001'))

        c = bitarray('110') * 3
        self.assertEQUAL(c, bitarray('110110110'))

        for a in self.randombitarrays():
            b = a.copy()
            for n in range(-3, 5):
                c = a * n
                self.assertEQUAL(c, bitarray(n * a.tolist()))
                c = n * a
                self.assertEqual(c, bitarray(n * a.tolist()))
                self.assertEQUAL(a, b)

        a = bitarray()
        self.assertRaises(TypeError, a.__mul__, None)

    def test_inplace_repeat(self):
        c = bitarray('1101110011')
        idc = id(c)
        c *= 0
        self.assertEQUAL(c, bitarray())
        self.assertEqual(idc, id(c))

        c = bitarray('110')
        c *= 3
        self.assertEQUAL(c, bitarray('110110110'))

        for a in self.randombitarrays():
            for n in range(-3, 5):
                b = a.copy()
                idb = id(b)
                b *= n
                self.assertEQUAL(b, bitarray(n * a.tolist()))
                self.assertEqual(idb, id(b))

        a = bitarray()
        self.assertRaises(TypeError, a.__imul__, None)

    def test_contains_simple(self):
        a = bitarray()
        self.assertFalse(False in a)
        self.assertFalse(True in a)
        self.assertTrue(bitarray() in a)
        a.append(True)
        self.assertTrue(True in a)
        self.assertFalse(False in a)
        a = bitarray([False])
        self.assertTrue(False in a)
        self.assertFalse(True in a)
        a.append(True)
        self.assertTrue(0 in a)
        self.assertTrue(1 in a)

    def test_contains_errors(self):
        a = bitarray()
        self.assertEqual(a.__contains__(1), False)
        a.append(1)
        self.assertEqual(a.__contains__(1), True)
        a = bitarray('0011')
        self.assertEqual(a.__contains__(bitarray('01')), True)
        self.assertEqual(a.__contains__(bitarray('10')), False)
        self.assertRaises((ValueError, TypeError), a.__contains__, 'asdf')
        self.assertRaises(ValueError, a.__contains__, 2)
        self.assertRaises(ValueError, a.__contains__, -1)

    def test_contains_range(self):
        for n in range(2, 50):
            a = bitarray(n)
            a.setall(0)
            self.assertTrue(False in a)
            self.assertFalse(True in a)
            a[randint(0, n - 1)] = 1
            self.assertTrue(True in a)
            self.assertTrue(False in a)
            a.setall(1)
            self.assertTrue(True in a)
            self.assertFalse(False in a)
            a[randint(0, n - 1)] = 0
            self.assertTrue(True in a)
            self.assertTrue(False in a)

    def test_contains_explicit(self):
        a = bitarray('011010000001')
        for s, r in [('', True), ('1', True), ('11', True), ('111', False),
                     ('011', True), ('0001', True), ('00011', False)]:
            self.assertEqual(bitarray(s) in a, r)


tests.append(SequenceMethodsTests)


# ---------------------------------------------------------------------------
class NumberMethodsTests(unittest.TestCase, Util):
    def test_misc(self):
        for a in self.randombitarrays():
            b = ~a
            c = a & b
            self.assertEqual(c.any(), False)
            self.assertEqual(a, a ^ c)
            d = a ^ b
            self.assertEqual(d.all(), True)
            b &= d
            self.assertEqual(~b, a)

    def test_size(self):
        a = bitarray('11001')
        b = bitarray('100111')
        self.assertRaises(ValueError, a.__and__, b)
        for x in a.__or__, a.__xor__, a.__iand__, a.__ior__, a.__ixor__:
            self.assertRaises(ValueError, x, b)

    def test_and(self):
        a = bitarray('11001')
        b = bitarray('10011')
        self.assertEQUAL(a & b, bitarray('10001'))

        b = bitarray('1001')
        self.assertRaises(ValueError, a.__and__, b)  # not same length

    def test_iand(self):
        a = bitarray('110010110')
        ida = id(a)
        a &= bitarray('100110011')
        self.assertEQUAL(a, bitarray('100010010'))
        self.assertEqual(ida, id(a))

    def test_or(self):
        a = bitarray('11001')
        b = bitarray('10011')
        aa = a.copy()
        bb = b.copy()
        self.assertEQUAL(a | b, bitarray('11011'))
        self.assertEQUAL(a, aa)
        self.assertEQUAL(b, bb)

    def test_ior(self):
        a = bitarray('110010110')
        b = bitarray('100110011')
        bb = b.copy()
        a |= b
        self.assertEQUAL(a, bitarray('110110111'))
        self.assertEQUAL(b, bb)

    def test_xor(self):
        a = bitarray('11001')
        b = bitarray('10011')
        self.assertEQUAL(a ^ b, bitarray('01010'))

    def test_ixor(self):
        a = bitarray('110010110')
        a ^= bitarray('100110011')
        self.assertEQUAL(a, bitarray('010100101'))

    def test_invert(self):
        a = bitarray('11011')
        b = ~a
        self.assertEQUAL(b, bitarray('00100'))
        self.assertEQUAL(a, bitarray('11011'))
        self.assertFalse(a is b)

        for a in self.randombitarrays():
            b = bitarray(a)
            b.invert()
            # pylint:disable=consider-using-enumerate
            for i in range(len(a)):
                self.assertEqual(b[i], not a[i])
            self.check_obj(b)
            c = ~a
            self.assertEQUAL(c, b)
            self.check_obj(c)


tests.append(NumberMethodsTests)


# ---------------------------------------------------------------------------
class ExtendTests(unittest.TestCase, Util):
    def test_wrongArgs(self):
        a = bitarray()
        self.assertRaises(TypeError, a.extend)
        self.assertRaises(TypeError, a.extend, None)
        self.assertRaises(TypeError, a.extend, True)
        self.assertRaises(TypeError, a.extend, 24)
        self.assertRaises(ValueError, a.extend, '0011201')

    def test_bitarray(self):
        a = bitarray()
        a.extend(bitarray())
        self.assertEqual(a, bitarray())
        a.extend(bitarray('110'))
        self.assertEqual(a, bitarray('110'))
        a.extend(bitarray('1110'))
        self.assertEqual(a, bitarray('1101110'))

        for a in self.randombitarrays():
            for b in self.randombitarrays():
                c = bitarray(a)
                idc = id(c)
                c.extend(b)
                self.assertEqual(id(c), idc)
                self.assertEqual(c, a + b)

    def test_list(self):
        a = bitarray()
        a.extend([0, 1, 3, None, {}])
        self.assertEqual(a, bitarray('01100'))
        a.extend([True, False])
        self.assertEqual(a, bitarray('0110010'))

        for a in self.randomlists():
            for b in self.randomlists():
                c = bitarray(a)
                idc = id(c)
                c.extend(b)
                self.assertEqual(id(c), idc)
                self.assertEqual(c.tolist(), a + b)
                self.check_obj(c)

    def test_tuple(self):
        a = bitarray()
        a.extend((0, 1, 2, 0, 3))
        self.assertEqual(a, bitarray('01101'))

        for a in self.randomlists():
            for b in self.randomlists():
                c = bitarray(a)
                idc = id(c)
                c.extend(tuple(b))
                self.assertEqual(id(c), idc)
                self.assertEqual(c.tolist(), a + b)
                self.check_obj(c)

    def test_generator(self):
        def bar():
            for x in ('', '1', None, True, []):
                yield x
        a = bitarray()
        a.extend(bar())
        self.assertEqual(a, bitarray('01010'))

        for a in self.randomlists():
            for b in self.randomlists():
                def foo():
                    # pylint:disable=cell-var-from-loop
                    for e in b:
                        yield e
                c = bitarray(a)
                idc = id(c)
                c.extend(foo())
                self.assertEqual(id(c), idc)
                self.assertEqual(c.tolist(), a + b)
                self.check_obj(c)

    def test_iterator1(self):
        a = bitarray()
        a.extend(iter([3, 9, 0, 1, -2]))
        self.assertEqual(a, bitarray('11011'))

        for a in self.randomlists():
            for b in self.randomlists():
                c = bitarray(a)
                idc = id(c)
                c.extend(iter(b))
                self.assertEqual(id(c), idc)
                self.assertEqual(c.tolist(), a + b)
                self.check_obj(c)

    def test_iterator2(self):
        a = bitarray()
        a.extend(itertools.repeat(True, 23))
        self.assertEqual(a, bitarray(23 * '1'))

    def test_string01(self):
        a = bitarray()
        a.extend('0110111')
        self.assertEqual(a, bitarray('0110111'))

        for a in self.randomlists():
            for b in self.randomlists():
                c = bitarray(a)
                idc = id(c)
                c.extend(''.join(['0', '1'][x] for x in b))
                self.assertEqual(id(c), idc)
                self.assertEqual(c.tolist(), a + b)
                self.check_obj(c)

    def test_extend_self(self):
        a = bitarray()
        a.extend(a)
        self.assertEqual(a, bitarray())

        a = bitarray('1')
        a.extend(a)
        self.assertEqual(a, bitarray('11'))

        a = bitarray('110')
        a.extend(a)
        self.assertEqual(a, bitarray('110110'))

        for a in self.randombitarrays():
            b = bitarray(a)
            a.extend(a)
            self.assertEqual(a, b + b)


tests.append(ExtendTests)


# ---------------------------------------------------------------------------
class MethodTests(unittest.TestCase, Util):
    def test_append_simple(self):
        a = bitarray()
        a.append(True)
        a.append(False)
        a.append(False)
        self.assertEQUAL(a, bitarray('100'))
        a.append(0)
        a.append(1)
        a.append('1')
        self.assertEQUAL(a, bitarray('100011'))

    def test_append_random(self):
        for a in self.randombitarrays():
            aa = a.tolist()
            b = a
            b.append(1)
            self.assertTrue(a is b)
            self.check_obj(b)
            self.assertEQUAL(b, bitarray(aa + [1]))

    def test_insert(self):
        a = bitarray()
        b = a
        a.insert(0, True)
        self.assertTrue(a is b)
        self.assertEqual(a, bitarray('1'))
        self.assertRaises(TypeError, a.insert)
        self.assertRaises(TypeError, a.insert, None)

        for a in self.randombitarrays():
            aa = a.tolist()
            for _ in range(50):
                item = bool(randint(0, 1))
                pos = randint(-len(a) - 2, len(a) + 2)
                a.insert(pos, item)
                aa.insert(pos, item)
                self.assertEqual(a.tolist(), aa)
                self.check_obj(a)

    def test_index1(self):
        a = bitarray()
        for i in (True, False, 1, 0):
            self.assertRaises(ValueError, a.index, i)

        a = bitarray(100 * [False])
        self.assertRaises(ValueError, a.index, True)
        self.assertRaises(TypeError, a.index)
        self.assertRaises(TypeError, a.index, 1, 'a')
        self.assertRaises(TypeError, a.index, 1, 0, 'a')
        a[20] = a[22] = a[27] = 1
        self.assertEqual(a.index('1'), 20)
        self.assertEqual(a.index(1, 21), 22)
        self.assertEqual(a.index(1, 23), 27)
        self.assertEqual(a.index(1, 27), 27)
        self.assertEqual(a.index(1, -73), 27)
        self.assertRaises(ValueError, a.index, 1, 5, 17)
        self.assertRaises(ValueError, a.index, 1, 5, -83)
        self.assertRaises(ValueError, a.index, 1, 23, 27)
        self.assertRaises(ValueError, a.index, 1, 28)
        self.assertEqual(a.index(0), 0)

        a = bitarray(200 * [True])
        self.assertRaises(ValueError, a.index, False)
        a[173] = a[187] = 0
        self.assertEqual(a.index(False), 173)
        self.assertEqual(a.index(True), 0)

    def test_index2(self):
        for n in range(50):
            for m in range(n):
                a = bitarray(n)
                a.setall(0)
                self.assertRaises(ValueError, a.index, 1)
                a[m] = 1
                self.assertEqual(a.index(1), m)

                a.setall(1)
                self.assertRaises(ValueError, a.index, 0)
                a[m] = 0
                self.assertEqual(a.index(0), m)

    def test_index3(self):
        a = bitarray('00001000' '00000000' '0010000')
        self.assertEqual(a.index(1), 4)
        self.assertEqual(a.index(1, 1), 4)
        self.assertEqual(a.index(0, 4), 5)
        self.assertEqual(a.index(1, 5), 18)
        self.assertRaises(ValueError, a.index, 1, 5, 18)
        self.assertRaises(ValueError, a.index, 1, 19)

    def test_index4(self):
        a = bitarray('11110111' '11111111' '1101111')
        self.assertEqual(a.index(0), 4)
        self.assertEqual(a.index(0, 1), 4)
        self.assertEqual(a.index(1, 4), 5)
        self.assertEqual(a.index(0, 5), 18)
        self.assertRaises(ValueError, a.index, 0, 5, 18)
        self.assertRaises(ValueError, a.index, 0, 19)

    def test_index5(self):
        a = bitarray(2000)
        a.setall(0)
        for _ in range(3):
            a[randint(0, 1999)] = 1
        aa = a.tolist()
        for _ in range(100):
            start = randint(0, 2000)
            stop = randint(0, 2000)
            try:
                res1 = a.index(1, start, stop)
            except ValueError:
                res1 = None
            try:
                res2 = aa.index(1, start, stop)
            except ValueError:
                res2 = None
            self.assertEqual(res1, res2)

    def test_index6(self):
        for n in range(1, 50):
            a = bitarray(n)
            i = randint(0, 1)
            a.setall(i)
            for _ in range(randint(1, 4)):
                a[randint(0, n - 1)] = 1 - i
            aa = a.tolist()
            for _ in range(100):
                start = randint(-50, n + 50)
                stop = randint(-50, n + 50)
                try:
                    res1 = a.index(1 - i, start, stop)
                except ValueError:
                    res1 = None
                try:
                    res2 = aa.index(1 - i, start, stop)
                except ValueError:
                    res2 = None
                self.assertEqual(res1, res2)

    def test_count_basic(self):
        a = bitarray('10011')
        self.assertEqual(a.count(), 3)
        self.assertEqual(a.count(True), 3)
        self.assertEqual(a.count(False), 2)
        self.assertEqual(a.count(1), 3)
        self.assertEqual(a.count(0), 2)
        self.assertEqual(a.count('1'), 3)
        self.assertEqual(a.count('0'), 2)
        self.assertRaises(TypeError, a.count, 0, 'A')
        self.assertRaises(TypeError, a.count, 0, 0, 'A')

    def test_count_byte(self):
        def count(n):  # count 1 bits in number
            cnt = 0
            while n:
                cnt += n & 1
                n >>= 1
            return cnt

        for i in range(256):
            a = bitarray()
            a.frombytes(bytes(bytearray([i])))
            self.assertEqual(len(a), 8)
            self.assertEqual(a.count(), count(i))
            self.assertEqual(a.count(), bin(i)[2:].count('1'))

    def test_count_whole_range(self):
        for a in self.randombitarrays():
            s = a.to01()
            self.assertEqual(a.count(1), s.count('1'))
            self.assertEqual(a.count(0), s.count('0'))

    def test_count_allones(self):
        n = 37
        a = bitarray(n)
        a.setall(1)
        for i in range(n):
            for j in range(i, n):
                self.assertEqual(a.count(1, i, j), j - i)

    def test_count_explicit(self):
        a = bitarray('01001100' '01110011' '01')
        self.assertEqual(a.count(), 9)
        self.assertEqual(a.count(0, 12), 3)
        self.assertEqual(a.count(1, -5), 3)
        self.assertEqual(a.count(1, 2, 17), 7)
        self.assertEqual(a.count(1, 6, 11), 2)
        self.assertEqual(a.count(0, 7, -3), 4)
        self.assertEqual(a.count(1, 1, -1), 8)
        self.assertEqual(a.count(1, 17, 14), 0)

    def test_count_random(self):
        for a in self.randombitarrays():
            s = a.to01()
            i = randint(-3, len(a) + 1)
            j = randint(-3, len(a) + 1)
            self.assertEqual(a.count(1, i, j), s[i:j].count('1'))
            self.assertEqual(a.count(0, i, j), s[i:j].count('0'))

    def test_search(self):
        a = bitarray('')
        self.assertEqual(a.search(bitarray('0')), [])
        self.assertEqual(a.search(bitarray('1')), [])

        a = bitarray('1')
        self.assertEqual(a.search(bitarray('0')), [])
        self.assertEqual(a.search(bitarray('1')), [0])
        self.assertEqual(a.search(bitarray('11')), [])

        a = bitarray(100 * '1')
        self.assertEqual(a.search(bitarray('0')), [])
        self.assertEqual(a.search(bitarray('1')), list(range(100)))

        a = bitarray('10010101110011111001011')
        for limit in range(10):
            self.assertEqual(a.search(bitarray('011'), limit),
                             [6, 11, 20][:limit])
        self.assertRaises(ValueError, a.search, bitarray())

    def test_itersearch(self):
        a = bitarray('10011')
        self.assertRaises(ValueError, a.itersearch, bitarray())
        self.assertRaises((TypeError, ValueError), a.itersearch, '')

        it = a.itersearch(bitarray('1'))
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)
        self.assertStopIteration(it)

    def test_search2(self):
        a = bitarray('10011')
        for s, res in [('0',     [1, 2]), ('1', [0, 3, 4]),
                       ('01',    [2]),    ('11', [3]),
                       ('000',   []),     ('1001', [0]),
                       ('011',   [2]),    ('0011', [1]),
                       ('10011', [0]),    ('100111', [])]:
            b = bitarray(s)
            self.assertEqual(a.search(b), res)
            # pylint:disable=unnecessary-comprehension
            self.assertEqual([p for p in a.itersearch(b)], res)

    def test_search3(self):
        a = bitarray('10010101110011111001011')
        for s, res in [('011', [6, 11, 20]),
                       ('111', [7, 12, 13, 14]),  # note the overlap
                       ('1011', [5, 19]),
                       ('100', [0, 9, 16])]:
            b = bitarray(s)
            self.assertEqual(a.search(b), res)
            self.assertEqual(list(a.itersearch(b)), res)
            # pylint:disable=unnecessary-comprehension
            self.assertEqual([p for p in a.itersearch(b)], res)

    def test_search4(self):
        for a in self.randombitarrays():
            aa = a.to01()
            for sub in '0', '1', '01', '01', '11', '101', '1111111':
                sr = a.search(bitarray(sub), 1)
                try:
                    p = sr[0]
                except IndexError:
                    p = -1
                self.assertEqual(p, aa.find(sub))

    def test_search_type(self):
        a = bitarray('10011')
        it = a.itersearch(bitarray('1'))
        self.assertIsInstance(type(it), type)

    def test_fill_simple(self):
        a = bitarray()
        self.assertEqual(a.fill(), 0)
        self.assertEqual(len(a), 0)

        a = bitarray('101')
        self.assertEqual(a.fill(), 5)
        self.assertEqual(a, bitarray('10100000'))
        self.assertEqual(a.fill(), 0)
        self.assertEqual(a, bitarray('10100000'))

    def test_fill_random(self):
        for a in self.randombitarrays():
            b = a.copy()
            res = b.fill()
            self.assertTrue(0 <= res < 8)
            self.check_obj(b)
            if len(a) % 8 == 0:
                self.assertEqual(b, a)
            else:
                self.assertTrue(len(b) % 8 == 0)
                self.assertNotEqual(b, a)
                self.assertEqual(b[:len(a)], a)
                self.assertEqual(b[len(a):],
                                 (len(b) - len(a)) * bitarray('0'))

    def test_invert_simple(self):
        a = bitarray()
        a.invert()
        self.assertEQUAL(a, bitarray())

        a = bitarray('11011')
        a.invert()
        self.assertEQUAL(a, bitarray('00100'))
        a.invert(2)
        self.assertEQUAL(a, bitarray('00000'))
        a.invert(-1)
        self.assertEQUAL(a, bitarray('00001'))

    def test_invert_errors(self):
        a = bitarray(5)
        self.assertRaises(IndexError, a.invert, 5)
        self.assertRaises(IndexError, a.invert, -6)
        self.assertRaises(TypeError, a.invert, "A")
        self.assertRaises(TypeError, a.invert, 0, 1)

    def test_invert_random(self):
        for a in self.randombitarrays(start=1):
            b = a.copy()
            c = a.copy()
            i = randint(0, len(a) - 1)
            b.invert(i)
            c[i] = not c[i]
            self.assertEQUAL(b, c)

    def test_sort_simple(self):
        a = bitarray('1101000')
        a.sort()
        self.assertEqual(a, bitarray('0000111'))

        a = bitarray('1101000')
        a.sort(reverse=True)
        self.assertEqual(a, bitarray('1110000'))
        a.sort(reverse=False)
        self.assertEqual(a, bitarray('0000111'))
        a.sort(True)
        self.assertEqual(a, bitarray('1110000'))
        a.sort(False)
        self.assertEqual(a, bitarray('0000111'))

        self.assertRaises(TypeError, a.sort, 'A')

    def test_sort_random(self):
        for rev in False, True:
            for a in self.randombitarrays():
                b = a.tolist()
                a.sort(rev)
                self.assertEqual(a, bitarray(sorted(b, reverse=rev)))

    def test_reverse_simple(self):
        # pylint:disable=implicit-str-concat
        for x, y in [('', ''), ('1', '1'), ('10', '01'), ('001', '100'),
                     ('1110', '0111'), ('11100', '00111'),
                     ('011000', '000110'), ('1101100', '0011011'),
                     ('11110000', '00001111'),
                     ('11111000011', '11000011111'),
                     ('11011111' '00100000' '000111',
                      '111000' '00000100' '11111011')]:
            a = bitarray(x)
            a.reverse()
            self.assertEQUAL(a, bitarray(y))

        self.assertRaises(TypeError, bitarray().reverse, 42)

    def test_reverse_random(self):
        for a in self.randombitarrays():
            b = a.copy()
            a.reverse()
            self.assertEQUAL(a, bitarray(b.tolist()[::-1]))
            self.assertEQUAL(a, b[::-1])

    def test_tolist(self):
        a = bitarray()
        self.assertEqual(a.tolist(), [])

        a = bitarray('110')
        self.assertEqual(a.tolist(), [True, True, False])
        self.assertEqual(a.tolist(True), [1, 1, 0])

        for as_ints in 0, 1:
            for elt in a.tolist(as_ints):
                self.assertIsInstance(elt, int if as_ints else bool)

        for lst in self.randomlists():
            a = bitarray(lst)
            self.assertEqual(a.tolist(), lst)

    def test_remove(self):
        a = bitarray('1010110')
        for val, res in [(False, '110110'), (True, '10110'),
                         (1, '0110'), (1, '010'), (0, '10'),
                         (0, '1'), (1, '')]:
            a.remove(val)
            self.assertEQUAL(a, bitarray(res))

        a = bitarray('0010011')
        b = a
        b.remove('1')
        self.assertTrue(b is a)
        self.assertEQUAL(b, bitarray('000011'))

    def test_remove_errors(self):
        a = bitarray()
        for i in (True, False, 1, 0):
            self.assertRaises(ValueError, a.remove, i)

        a = bitarray(21)
        a.setall(0)
        self.assertRaises(ValueError, a.remove, 1)
        a.setall(1)
        self.assertRaises(ValueError, a.remove, 0)

    def test_pop_simple(self):
        for x, n, r, y in [('1', 0, True, ''),
                           ('0', -1, False, ''),
                           ('0011100', 3, True, '001100')]:
            a = bitarray(x)
            self.assertEqual(a.pop(n), r)
            self.assertEqual(a, bitarray(y))

        a = bitarray('01')
        self.assertEqual(a.pop(), True)
        self.assertEqual(a.pop(), False)
        self.assertRaises(IndexError, a.pop)

    def test_pop_random(self):
        for a in self.randombitarrays():
            self.assertRaises(IndexError, a.pop, len(a))
            self.assertRaises(IndexError, a.pop, -len(a) - 1)
            if len(a) == 0:
                continue
            aa = a.tolist()
            self.assertEqual(a.pop(), aa[-1])
            self.check_obj(a)

        for a in self.randombitarrays(start=1):
            n = randint(-len(a), len(a) - 1)
            aa = a.tolist()
            self.assertEqual(a.pop(n), aa[n])
            aa.pop(n)
            self.assertEqual(a, bitarray(aa))
            self.check_obj(a)

    def test_clear(self):
        for a in self.randombitarrays():
            ida = id(a)
            a.clear()
            self.assertEqual(a, bitarray())
            self.assertEqual(id(a), ida)
            self.assertEqual(len(a), 0)

    def test_setall(self):
        a = bitarray(5)
        a.setall(True)
        self.assertEQUAL(a, bitarray('11111'))
        a.setall(False)
        self.assertEQUAL(a, bitarray('00000'))

    def test_setall_empty(self):
        a = bitarray()
        for v in 0, 1:
            a.setall(v)
            self.assertEQUAL(a, bitarray())

    def test_setall_random(self):
        for a in self.randombitarrays():
            val = randint(0, 1)
            b = a
            b.setall(val)
            self.assertEqual(b, bitarray(len(b) * [val]))
            self.assertTrue(a is b)
            self.check_obj(b)

    def test_bytereverse_explicit(self):
        # pylint:disable=implicit-str-concat
        for x, y in [('', ''),
                     ('1', '0'),
                     ('1011', '0000'),
                     ('111011', '001101'),
                     ('11101101', '10110111'),
                     ('000000011', '100000000'),
                     ('11011111' '00100000' '000111',
                      '11111011' '00000100' '001110')]:
            a = bitarray(x)
            a.bytereverse()
            self.assertEqual(a, bitarray(y))

    def test_bytereverse_byte(self):
        for i in range(256):
            a = bitarray()
            a.frombytes(bytes(bytearray([i])))
            b = a.copy()
            b.bytereverse()
            self.assertEqual(b, a[::-1])
            self.check_obj(b)


tests.append(MethodTests)


# ---------------------------------------------------------------------------
class BytesTests(unittest.TestCase, Util):
    # pylint:disable=no-self-use
    # noinspection PyMethodMayBeStatic
    def randombytes(self):
        for n in range(1, 20):
            yield os.urandom(n)

    def test_frombytes_simple(self):
        a = bitarray()
        a.frombytes(b'A')
        self.assertEqual(a, bitarray('01000001'))

        b = a
        b.frombytes(b'BC')
        self.assertEQUAL(b, bitarray('01000001' '01000010' '01000011'))
        self.assertTrue(b is a)

    def test_frombytes_empty(self):
        for a in self.randombitarrays():
            b = a.copy()
            a.frombytes(b'')
            self.assertEQUAL(a, b)
            self.assertFalse(a is b)

    def test_frombytes_errors(self):
        a = bitarray()
        self.assertRaises(TypeError, a.frombytes)
        self.assertRaises(TypeError, a.frombytes, b'', b'')
        self.assertRaises(TypeError, a.frombytes, 1)

    def test_frombytes_random(self):
        for b in self.randombitarrays():
            for s in self.randombytes():
                a = bitarray()
                a.frombytes(s)
                c = b.copy()
                b.frombytes(s)
                self.assertEQUAL(b[-len(a):], a)
                self.assertEQUAL(b[:-len(a)], c)
                self.assertEQUAL(b, c + a)

    def test_tobytes_empty(self):
        a = bitarray()
        self.assertEqual(a.tobytes(), b'')

    def test_tobytes_explicit_ones(self):
        for n, s in [(1, b'\x80'), (2, b'\xc0'), (3, b'\xe0'), (4, b'\xf0'),
                     (5, b'\xf8'), (6, b'\xfc'), (7, b'\xfe'), (8, b'\xff'),
                     (12, b'\xff\xf0'), (15, b'\xff\xfe'), (16, b'\xff\xff'),
                     (17, b'\xff\xff\x80'), (24, b'\xff\xff\xff')]:
            a = bitarray(n)
            a.setall(1)
            self.assertEqual(a.tobytes(), s)

    def test_unpack_simple(self):
        a = bitarray('01')
        self.assertIsInstance(a.unpack(), bytes)
        self.assertEqual(a.unpack(), b'\x00\xff')
        self.assertEqual(a.unpack(b'A'), b'A\xff')
        self.assertEqual(a.unpack(b'0', b'1'), b'01')
        self.assertEqual(a.unpack(one=b'\x01'), b'\x00\x01')
        self.assertEqual(a.unpack(zero=b'A'), b'A\xff')
        self.assertEqual(a.unpack(one=b't', zero=b'f'), b'ft')

    def test_unpack_random(self):
        for a in self.randombitarrays():
            self.assertEqual(a.unpack(b'0', b'1'), a.to01().encode())
            # round trip
            b = bitarray()
            b.pack(a.unpack())
            self.assertEqual(b, a)
            # round trip with invert
            b = bitarray()
            b.pack(a.unpack(b'\x01', b'\x00'))
            b.invert()
            self.assertEqual(b, a)

    def test_unpack_errors(self):
        a = bitarray('01')
        self.assertRaises(TypeError, a.unpack, b'')
        self.assertRaises(TypeError, a.unpack, b'0', b'')
        self.assertRaises(TypeError, a.unpack, b'a', zero=b'b')
        self.assertRaises(TypeError, a.unpack, foo=b'b')
        self.assertRaises(TypeError, a.unpack, one=b'aa', zero=b'b')

        self.assertRaises(TypeError, a.unpack, '0')
        self.assertRaises(TypeError, a.unpack, one='a')
        self.assertRaises(TypeError, a.unpack, b'0', '1')

    def test_pack_simple(self):
        a = bitarray()
        a.pack(b'\x00')
        self.assertEqual(a, bitarray('0'))
        a.pack(b'\xff')
        self.assertEqual(a, bitarray('01'))
        a.pack(b'\x01\x00\x7a')
        self.assertEqual(a, bitarray('01101'))

    def test_pack_random(self):
        a = bitarray()
        for n in range(256):
            a.pack(bytes(bytearray([n])))
        self.assertEqual(a, bitarray('0' + 255 * '1'))

    def test_pack_errors(self):
        a = bitarray()
        self.assertRaises(TypeError, a.pack, 0)
        self.assertRaises(TypeError, a.pack, '1')
        self.assertRaises(TypeError, a.pack, [1, 3])
        self.assertRaises(TypeError, a.pack, bitarray())


tests.append(BytesTests)


# ---------------------------------------------------------------------------
class FileTests(unittest.TestCase, Util):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmpfname = os.path.join(self.tmpdir, 'testfile')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def read_file(self):
        with open(self.tmpfname, 'rb') as fi:
            return fi.read()

    def assertFileSize(self, size):
        self.assertEqual(os.path.getsize(self.tmpfname), size)

    def test_pickle(self):
        for a in self.randombitarrays():
            with open(self.tmpfname, 'wb') as fo:
                pickle.dump(a, fo)
            with open(self.tmpfname, 'rb') as fi:
                b = pickle.load(fi)
            self.assertFalse(b is a)
            self.assertEQUAL(a, b)

    def test_shelve(self):
        if not shelve or hasattr(sys, 'gettotalrefcount'):
            return

        d = shelve.open(self.tmpfname)
        stored = []
        for a in self.randombitarrays():
            key = hashlib.md5(repr(a).encode()).hexdigest()
            d[key] = a
            stored.append((key, a))
        d.close()
        del d

        d = shelve.open(self.tmpfname)
        for k, v in stored:
            # noinspection PyTypeChecker
            self.assertEQUAL(d[k], v)
        d.close()

    def test_fromfile_empty(self):
        with open(self.tmpfname, 'wb') as _:
            pass
        self.assertFileSize(0)

        a = bitarray()
        with open(self.tmpfname, 'rb') as fi:
            a.fromfile(fi)
        self.assertEqual(a, bitarray())

    def test_fromfile_Foo(self):
        with open(self.tmpfname, 'wb') as fo:
            fo.write(b'Foo')
        self.assertFileSize(3)

        a = bitarray()
        with open(self.tmpfname, 'rb') as fi:
            a.fromfile(fi)
        self.assertEqual(a, bitarray('01000110' '01101111' '01101111'))

    def test_fromfile_wrong_args(self):
        a = bitarray()
        self.assertRaises(TypeError, a.fromfile)
        # self.assertRaises(TypeError, a.fromfile, StringIO())  # file not open
        self.assertRaises(Exception, a.fromfile, 42)
        self.assertRaises(Exception, a.fromfile, 'bar')

        with open(self.tmpfname, 'wb') as _:
            pass
        with open(self.tmpfname, 'rb') as fi:
            self.assertRaises(TypeError, a.fromfile, fi, None)

    def test_fromfile_errors(self):
        with open(self.tmpfname, 'wb') as fo:
            fo.write(b'0123456789')
        self.assertFileSize(10)

        a = bitarray()
        with open(self.tmpfname, 'wb') as fi:
            self.assertRaises(Exception, a.fromfile, fi)

        with open(self.tmpfname, 'r') as fi:  # pylint:disable=unspecified-encoding
            self.assertRaises(TypeError, a.fromfile, fi)

    def test_from_large_files(self):
        for n in range(65534, 65538):
            data = os.urandom(n)
            with open(self.tmpfname, 'wb') as fo:
                fo.write(data)

            a = bitarray()
            with open(self.tmpfname, 'rb') as fi:
                a.fromfile(fi)
            self.assertEqual(len(a), 8 * n)
            self.assertEqual(a.buffer_info()[1], n)
            self.assertEqual(a.tobytes(), data)

    def test_fromfile_extend_existing(self):
        with open(self.tmpfname, 'wb') as fo:
            fo.write(b'Foo')

        foo = '010001100110111101101111'
        a = bitarray('1')
        with open(self.tmpfname, 'rb') as fi:
            a.fromfile(fi)
        self.assertEqual(a, bitarray('1' + foo))

        for n in range(20):
            a = bitarray(n)
            a.setall(1)
            with open(self.tmpfname, 'rb') as fi:
                a.fromfile(fi)
            self.assertEqual(a, bitarray(n * '1' + foo))

    def test_fromfile_n(self):
        a = bitarray()
        a.frombytes(b'ABCDEFGHIJ')
        with open(self.tmpfname, 'wb') as fo:
            a.tofile(fo)
        self.assertFileSize(10)

        with open(self.tmpfname, 'rb') as f:
            a = bitarray()

            a.fromfile(f, 0)
            self.assertEqual(a.tobytes(), b'')

            a.fromfile(f, 1)
            self.assertEqual(a.tobytes(), b'A')

            f.read(1)  # skip B
            a.fromfile(f, 1)
            self.assertEqual(a.tobytes(), b'AC')

            a = bitarray()
            a.fromfile(f, 2)
            self.assertEqual(a.tobytes(), b'DE')

            a.fromfile(f, 1)
            self.assertEqual(a.tobytes(), b'DEF')

            a.fromfile(f, 0)
            self.assertEqual(a.tobytes(), b'DEF')

            a.fromfile(f)
            self.assertEqual(a.tobytes(), b'DEFGHIJ')

            a.fromfile(f)
            self.assertEqual(a.tobytes(), b'DEFGHIJ')

        a = bitarray()
        with open(self.tmpfname, 'rb') as f:
            f.read(1)
            self.assertRaises(EOFError, a.fromfile, f, 10)
        # check that although we received an EOFError, the bytes were read
        self.assertEqual(a.tobytes(), b'BCDEFGHIJ')

        a = bitarray()
        with open(self.tmpfname, 'rb') as f:
            # negative values - like omitting the argument
            a.fromfile(f, -1)
            self.assertEqual(a.tobytes(), b'ABCDEFGHIJ')
            self.assertRaises(EOFError, a.fromfile, f, 1)

    def test_fromfile_BytesIO(self):
        f = BytesIO(b'somedata')
        a = bitarray()
        a.fromfile(f, 4)
        self.assertEqual(len(a), 32)
        self.assertEqual(a.tobytes(), b'some')
        a.fromfile(f)
        self.assertEqual(len(a), 64)
        self.assertEqual(a.tobytes(), b'somedata')

    def test_tofile_empty(self):
        a = bitarray()
        with open(self.tmpfname, 'wb') as f:
            a.tofile(f)

        self.assertFileSize(0)

    def test_tofile_Foo(self):
        a = bitarray('0100011' '001101111' '01101111')
        b = a.copy()
        with open(self.tmpfname, 'wb') as f:
            a.tofile(f)
        self.assertEQUAL(a, b)

        self.assertFileSize(3)
        self.assertEqual(self.read_file(), b'Foo')

    def test_tofile_random(self):
        for a in self.randombitarrays():
            with open(self.tmpfname, 'wb') as fo:
                a.tofile(fo)
            n = bits2bytes(len(a))
            self.assertFileSize(n)
            raw = self.read_file()
            self.assertEqual(len(raw), n)
            self.assertEqual(raw, a.tobytes())

    def test_tofile_errors(self):
        n = 100
        a = bitarray(8 * n)
        self.assertRaises(TypeError, a.tofile)

        with open(self.tmpfname, 'wb') as f:
            a.tofile(f)
        self.assertFileSize(n)
        # write to closed file
        self.assertRaises(ValueError, a.tofile, f)

        with open(self.tmpfname, 'w') as f:  # pylint:disable=unspecified-encoding
            self.assertRaises(TypeError, a.tofile, f)

        with open(self.tmpfname, 'rb') as f:
            self.assertRaises(Exception, a.tofile, f)

    def test_tofile_large(self):
        n = 100 * 1000
        a = bitarray(8 * n)
        a.setall(0)
        a[2::37] = 1
        with open(self.tmpfname, 'wb') as f:
            a.tofile(f)
        self.assertFileSize(n)

        raw = self.read_file()
        self.assertEqual(len(raw), n)
        self.assertEqual(raw, a.tobytes())

    def test_tofile_ones(self):
        for n in range(20):
            a = n * bitarray('1')
            with open(self.tmpfname, 'wb') as fo:
                a.tofile(fo)

            raw = self.read_file()
            self.assertEqual(len(raw), bits2bytes(len(a)))
            # when we fill the unused bits in a, we can compare
            a.fill()
            b = bitarray()
            b.frombytes(raw)
            self.assertEqual(a, b)

    def test_tofile_BytesIO(self):
        for n in list(range(10)) + list(range(65534, 65538)):
            data = os.urandom(n)
            a = bitarray(0)
            a.frombytes(data)
            self.assertEqual(len(a), 8 * n)
            f = BytesIO()
            a.tofile(f)
            self.assertEqual(f.getvalue(), data)


tests.append(FileTests)


# -------------------------- Buffer Interface -------------------------------
class BufferInterfaceTests(unittest.TestCase):
    def test_read_simple(self):
        a = bitarray('01000001' '01000010' '01000011')
        v = memview(a)
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 65)
        self.assertEqual(v.tobytes(), b'ABC')
        a[13] = 1
        self.assertEqual(v.tobytes(), b'AFC')

    def test_read_random(self):
        a = bitarray()
        a.frombytes(os.urandom(100))
        v = memview(a)
        self.assertEqual(len(v), 100)
        b = a[34 * 8: 67 * 8]
        self.assertEqual(v[34:67].tobytes(), b.tobytes())
        self.assertEqual(v.tobytes(), a.tobytes())

    def test_resize(self):
        a = bitarray('01000001' '01000010' '01000011')
        v = memview(a)
        self.assertRaises(BufferError, a.append, 1)
        self.assertRaises(BufferError, a.clear)
        self.assertRaises(BufferError, a.__delitem__, slice(0, 8))
        self.assertEqual(v.tobytes(), a.tobytes())

    def test_write(self):
        a = bitarray(8000)
        a.setall(0)
        v = memview(a)
        self.assertFalse(v.readonly)
        # noinspection PyTypeChecker
        v[500] = 255
        self.assertEqual(a[3999:4009], bitarray('0111111110'))
        a[4003] = 0
        self.assertEqual(a[3999:4009], bitarray('0111011110'))
        v[301:304] = b'ABC'
        self.assertEqual(a[300 * 8: 305 * 8].tobytes(), b'\x00ABC\x00')

    # noinspection PyTypeChecker
    def test_write_py3(self):
        a = bitarray(40)
        a.setall(0)
        m = memview(a)
        v = m[1:4]
        v[0] = 65
        v[1] = 66
        v[2] = 67
        self.assertEqual(a.tobytes(), b'\x00ABC\x00')


tests.append(BufferInterfaceTests)


# ---------------------------------------------------------------------------
class TestsFrozenbitarray(unittest.TestCase, Util):
    def test_init(self):
        a = frozenbitarray('110')
        self.assertEqual(a, bitarray('110'))
        self.assertEqual(a.to01(), '110')

    def test_methods(self):
        # test a few methods which do not raise the TypeError
        a = frozenbitarray('1101100')
        self.assertEqual(a[2], 0)
        self.assertEqual(a[:4].to01(), '1101')
        self.assertEqual(a.count(), 4)
        self.assertEqual(a.index(0), 2)
        b = a.copy()
        self.assertEqual(b, a)
        self.assertTrue(repr(type(b)).lower().rstrip('>').rstrip("'").endswith('bitarray.frozenbitarray'))
        self.assertEqual(len(b), 7)
        self.assertEqual(b.all(), False)
        self.assertEqual(b.any(), True)

    def test_init_from_bitarray(self):
        for a in self.randombitarrays():
            b = frozenbitarray(a)
            self.assertFalse(b is a)
            self.assertEqual(b, a)
            c = frozenbitarray(b)
            self.assertEqual(c, b)
            self.assertFalse(c is b)
            self.assertEqual(hash(c), hash(b))

    def test_repr(self):
        a = frozenbitarray()
        self.assertEqual(repr(a), "frozenbitarray()")
        self.assertEqual(str(a), "")
        a = frozenbitarray('10111')
        self.assertEqual(repr(a), "frozenbitarray('10111')")
        self.assertEqual(str(a), "10111")

    def test_immutable(self):
        a = frozenbitarray('111')
        self.assertRaises(TypeError, a.append, True)
        self.assertRaises(TypeError, a.clear)
        self.assertRaises(TypeError, a.__delitem__, 0)
        self.assertRaises(TypeError, a.__setitem__, 0, 0)

    def test_dictkey(self):
        a = frozenbitarray('01')
        b = frozenbitarray('1001')
        d = {a: 123, b: 345}
        self.assertEqual(d[frozenbitarray('01')], 123)
        self.assertEqual(d[frozenbitarray(b)], 345)

    def test_dictkey2(self):  # taken slightly modified from issue #74
        a1 = frozenbitarray([True, False])
        a2 = frozenbitarray([False, False])
        dct = {a1: "one", a2: "two"}
        a3 = frozenbitarray([True, False])
        self.assertEqual(a3, a1)
        self.assertEqual(dct[a3], 'one')

    def test_mix(self):
        a = bitarray('110')
        b = frozenbitarray('0011')
        self.assertEqual(a + b, bitarray('1100011'))
        a.extend(b)
        self.assertEqual(a, bitarray('1100011'))

    def test_pickle(self):
        for a in self.randombitarrays():
            f = frozenbitarray(a)
            g = pickle.loads(pickle.dumps(f))
            self.assertEqual(f, g)
            self.assertTrue(repr(g).startswith('frozenbitarray'))


tests.append(TestsFrozenbitarray)


# ---------------------------------------------------------------------------
def run(verbosity=1, repeat=1):
    print('bitarray is installed in: %s' % os.path.dirname(__file__))
    print('sys.version: %s' % sys.version)
    suite = unittest.TestSuite()
    for cls in tests:
        for _ in range(repeat):
            # case = unittest.makeSuite(cls)
            case = unittest.TestLoader().loadTestsFromTestCase(cls)
            suite.addTest(case)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    run()
