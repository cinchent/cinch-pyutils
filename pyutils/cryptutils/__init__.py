# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2016-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to provide supplemental encryption/decryption algorithms.
"""
import os
from pathlib import Path
import socket
import hashlib
import operator
from functools import reduce
from base64 import (urlsafe_b64decode as ub64dec, urlsafe_b64encode as ub64enc)
import string
import random

try:
    import getpass
    # noinspection PyPackageRequirements
    from cryptography.hazmat.primitives import serialization as crypto_serialization
    # noinspection PyPackageRequirements
    from cryptography.hazmat.primitives.asymmetric import rsa
    # noinspection PyPackageRequirements
    from cryptography.hazmat.backends import default_backend as crypto_default_backend
except ImportError:  # (tolerate during 'pyutils' install only)
    getpass = crypto_serialization = rsa = crypto_default_backend = None

try:
    # noinspection PyPackageRequirements
    from passlib.context import CryptContext
except ImportError:
    CryptContext = None

OBFUSCATOR_FILE = os.getenv('OBFUSCATOR_FILE', '~/.obfuscator')  # Canonical file containing obfuscator key
OBFUSKEY = os.getenv('OBFUSKEY')                                 # Obfuscator key (None => read from file)


# Standard hashes:
def sha512(plaintext):
    """ Returns sha512 of plaintext. """
    if CryptContext is None:
        raise NotImplementedError("Requires 'passlib' to be installed")
    # noinspection PyCallingNonCallable
    return CryptContext(['pbkdf2_sha512']).encrypt(plaintext)


def md5(plaintext):
    """ Returns md5sum of plaintext. """
    return hashlib.md5(plaintext.encode()).hexdigest()


# pylint:disable=invalid-name
class Mersenne:
    """
    Mersenne Twister pseudo-random number generator, unique and spanning
    to 32 bits.

    (Adapted from James Katz: https://github.com/james727/MTP)

    Other references:
     * http://www.eetimes.com/document.asp?doc_id=1274550
       (tutorial on linear feedback shift registers)
     * http://www.quadibloc.com/crypto/jscrypt.htm
       (the Mersenne Twister and LFSRs in particular)
     * https://en.wikipedia.org/wiki/Mersenne_Twister
       (background information and pseudocode)
    """

    def __init__(self, seed=5489):
        """ It's a twister! """
        # Constants:
        self.upper_mask = 1 << 31
        self.lower_mask = self.upper_mask - 1

        self.f = 1812433253   # (initialization factor)
        self.us = 11          # (upper-order shift)
        self.ss = 7           # (mid-upper shift)
        self.bm = 0x9D2C5680  # (mid-upper mask)
        self.ts = 15          # (mid-lower shift)
        self.cm = 0xEFC60000  # (mid-lower mask)
        self.ls = 18          # (lower-order shift)
        self.tf = 0x9908B0DF  # (temp flipmask)
        self.m = 397          # (pail modulus)
        self.n = 624          # (# bins)

        # State bins:
        self.state = [0] * self.n
        self.index = self.n

        # Seed state bins initially.
        self.state[0] = seed
        for i in range(1, self.n):
            prev = self.state[i - 1]
            self.state[i] = self.uint32(self.f * (prev ^ (prev >> 30)) + i)

    @staticmethod
    def uint32(number):
        """ Utility: Masks integer to limit to 32 bits. """
        return int(0xFFFFFFFF & number)

    def __twist(self):
        """ Let's do the twist. """
        for i in range(self.n):
            tmp = self.uint32((self.state[i] & self.upper_mask) +
                              (self.state[(i + 1) % self.n] & self.lower_mask))
            tmp_shift = tmp >> 1
            if tmp % 2 != 0:
                tmp_shift ^= self.tf
            self.state[i] = self.state[(i + self.m) % self.n] ^ tmp_shift
        self.index = 0

    def rand(self):
        """ Returns pseudo-random number in 32-bit range. """
        if self.index >= self.n:
            self.__twist()
        y = self.state[self.index]
        y ^= y >> self.us
        y ^= (y << self.ss) & self.bm
        y ^= (y << self.ts) & self.cm
        y ^= y >> self.ls
        self.index += 1
        return self.uint32(y)


class TextObfuscator:
    """ Simple text obfuscator. """
    _pairs = (''.join(sorted(string.ascii_letters)) + string.digits +
              '-_=', string.printable[:62] + '_-.')

    @classmethod
    def _xlat(cls, text, table=None):
        """ Helper: Text translation from table (default: _pairs). """
        return text.translate(str.maketrans(*(table or cls._pairs)))

    @staticmethod
    def _scram(text, n):
        """ Text scrambler. """
        return reduce(operator.add, [text[i::n] for i in range(n)])

    @staticmethod
    def _rot(text, n):
        """ Text rotator. """
        return text[n:] + text[:n]

    @staticmethod
    def _inv(text, xor):
        """ Text inverter. """
        return ''.join([chr(ord(text[i]) ^ ord(xor[i % len(xor)]))
                        for i in range(len(text))]) if text and xor else text

    @classmethod
    def encode_text(cls, plaintext, seed='', key=''):
        """ Trivially encodes plaintext; seed to uniquify, key to encrypt. """
        txb = bytes(cls._inv(md5(seed) + plaintext, cls._rot(md5(key), 11)),
                    encoding='utf-8')
        return cls._scram(cls._xlat(ub64enc(txb).decode()[::-1]), 4)

    @classmethod
    def decode_text(cls, ciphtext, key=''):
        """ Trivially decrypts/decodes encoded text. """
        xlt = cls._scram(ciphtext, len(ciphtext) // 4)
        return cls._inv(ub64dec(cls._xlat(xlt, table=cls._pairs[::-1])[::-1])
                        .decode(), cls._rot(md5(key), 11))[32:]

    @classmethod
    def decode_file(cls, filespec, key=''):
        """ Trivially decrypts/decodes encoded text from file. """
        with open(filespec, "rt", encoding='utf-8') as fd:
            result = cls.decode_text(fd.read(), key=key)
        return result


class NumericObfuscator:
    """ Simple integer obfuscator, with integrity-checking. """
    def __init__(self, seed=None):
        """ Numeric obfuscator. """
        if seed is None:
            seed = random.getrandbits(32)
        self.__twister = Mersenne(seed)  # (seed the twister)
        self.__pails = tuple((self._tuplify(self.__twister.rand())
                              for _r in range(10)))

    @staticmethod
    def _hexits(width):
        """ Hex digits adjustor."""
        return min(width - (width > 5), 8)

    @staticmethod
    def _tuplify(n, digits=8, base=16):
        """
        Utility: Converts numeric value to decomposition tuple of base-specific
        decimal digit values.
        """
        return tuple(((n // base**i) % base for i in range(digits)))

    @staticmethod
    def _squash(t, base=16):
        """
        Utility: Converts sequence of base-specific decimal digit values
        to numeric value.
        """
        return sum((d * base**i) for i, d in enumerate(t))

    def __flip(self, inp, r):
        """ Flips value according to a Mersenne twister. """
        pailn = r % len(self.__pails)
        return tuple(d ^ self.__pails[pailn][c] for c, d in enumerate(inp))

    # noinspection PyArgumentList,PyShadowingBuiltins
    def encode(self, inp, width=10, pail=None):
        """
        Encodes an integer to a different integer value.

        :param inp:   Nonnegative integer to encode
        :type  inp:   int
        :param width: Max # decimal digits for input integer
        :type  width: int
        :param pail:  Fixed "pail" to use for encoding (0-9; None => random)
        :type  pail:  Union(int, None)

        :return: Result:
                  [0]: Encoded integer    \\__ Sum these to represent the
                  [1]: Encoding metadata  //   encoding in a single value
        :rtype:  tuple

        .. note::
         * If the input integer is negative or larger than `width`, the
           output is -1, meaning "invalid".
         * Range of the output encoded integer (separate from the output
           metadata) will be one more decimal digit than the input integer.
         * Specifying a `pail` parameter will cause the output integer
           to be deterministic; otherwise, many different valid output
           encodings will typically result given the same input integer.
         * The returned metadata is needed to decode the encoded integer;
           the result tuple can be combined non-destructively by summation,
           and passed to the :method:`decode()` with `meta` specified as None.
        """
        if 0 <= inp < 10 ** (width + 1):
            if pail is None:
                # @@@@@ pail = random.randint(0, len(self.__pails) - 1)
                pail = self.__twister.rand() % len(self.__pails)
            out = self._squash(self.__flip(
                self._tuplify(inp, digits=self._hexits(width)), pail))
            result = out, (sum(self._tuplify(out, base=10, digits=width + 1))
                           % 10 * 10 + pail % 10) * 10 ** (width + 1)
        else:
            result = (-1, 0)
        return result

    # noinspection PyArgumentList
    def decode(self, inp, meta=None, width=10, check=True):
        """
        Decodes a previously encoded integer to its original value.

        :param inp:   Encoded integer (see :method:`encode()')
        :type  inp:   int
        :param meta:  Encoded metadata (None => extract from high-order
                      decimal digits of `inp`)
        :type  meta:  Union(int, None)
        :param width: Max # decimal digits for decoded integer (must match
                      width used for encoding)
        :type  width: int
        :param check: "Check validity of the specified encoded integer."
        :type  check: bool

        :return: Decoded integer (None => invalid encoded integer, if `check`)
        :rtype:  Union(int, None)

        .. note::
         * `meta` can alternatively be specified as the same fixed `pail` value
           passed to :method:`encode()`
        """
        widmod = 10 ** (width + 1)
        pfx, inp = divmod(inp, widmod)
        if meta is None:
            meta = pfx
        elif not meta % 100:
            meta //= widmod
        pfx = meta // 100
        cksum, meta = divmod(meta, 10)
        if check and (sum(self._tuplify(inp, base=10, digits=width + 1))
                      % 10 + pfx != cksum):
            out = None
        else:
            out = self._squash(self.__flip(
                self._tuplify(inp, digits=self._hexits(width)), meta))
        return out


def generate_keys(keydir=None, keyfile_name=None, username=None, hostname=None):
    """
    Generates an RSA key pair and (optionally) writes the public and private
    keys to separate files.

    :param keydir:       Directory where to store key files, created if absent
                         when writing files (None => current directory)
    :type  keydir:       Union(str, None)
    :param keyfile_name: Base filename for file pair, sans extension
                         (function appends applicable extension;
                          None => don't write files)
    :type  keyfile_name: Union(str, None)
    :param username:     Username for public key label (None => current user)
    :type  username:     Union(str, None)
    :param hostname:     Hostname for public key label (None => this host)
    :type  hostname:     Union(str, None)

    :return: Result pair:
              [0]: private key (PEM representation)
              [1]: public key (base64-encoded, suffixed with username/hostname label)
    :rtype:  tuple

    .. note::
     * Partially provides the functionality of the OpenSSH 'ssh-keygen' utility,
       limited to 2048-bit RSA keys.
    """
    ext_publ, ext_priv = ('.pem', '.key')
    key = rsa.generate_private_key(backend=crypto_default_backend(),
                                   public_exponent=65537,
                                   key_size=2048)

    # noinspection PyTypeChecker
    private_key = (key.private_bytes(crypto_serialization.Encoding.PEM,
                                     crypto_serialization.PrivateFormat.PKCS8,
                                     crypto_serialization.NoEncryption()).decode()
                   .replace("PRIVATE KEY", "RSA PRIVATE KEY"))  # (for 'ssh-keygen' compatibility)
    # noinspection PyTypeChecker
    public_key = key.public_key().public_bytes(crypto_serialization.Encoding.OpenSSH,
                                               crypto_serialization.PublicFormat.OpenSSH).decode()

    if not username:
        username = getpass.getuser()
    if not hostname:
        hostname = socket.gethostname()

    if keyfile_name:
        if keydir:
            os.makedirs(keydir, exist_ok=True)
        base_filespec = os.path.join(keydir or os.path.curdir, keyfile_name)
        Path(base_filespec).with_suffix(ext_priv).write_text(private_key, encoding='utf-8')
        Path(base_filespec).with_suffix(ext_publ).write_text(
            "{} {}@{}\n".format(public_key.strip(), username, hostname), encoding='utf-8')

    return private_key, public_key
