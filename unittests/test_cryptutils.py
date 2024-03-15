"""
Tests for cryptutils
"""
# pylint:disable=missing-class-docstring,missing-function-docstring,wrong-import-position

import unittest

try:
    # pylint:disable=import-error
    # noinspection PyUnresolvedReferences
    from cryptutils import (TextObfuscator, NumericObfuscator)
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from pyutils.cryptutils import (TextObfuscator, NumericObfuscator)


class TestsTextObfuscator(unittest.TestCase):
    def test_simple(self):
        ptxt = "now is the time"
        ctxt = TextObfuscator.encode_text(ptxt)
        self.assertEqual(TextObfuscator.decode_text(ctxt), ptxt)

    def test_seed(self):
        ptxt = "now is the time"
        ctxt = TextObfuscator.encode_text(ptxt, seed="johnny apple")
        self.assertEqual(TextObfuscator.decode_text(ctxt), ptxt)

    def test_key(self):
        ptxt = "now is the time"
        key = "run, forrest, run"
        ctxt = TextObfuscator.encode_text(ptxt, key=key)
        self.assertEqual(TextObfuscator.decode_text(ctxt, key=key), ptxt)
        self.assertNotEqual(TextObfuscator.decode_text(ctxt), ptxt)

    def test_seed_key(self):
        ptxt = "now is the time"
        key = "run, forrest, run"
        ctxt = TextObfuscator.encode_text(ptxt, key=key, seed="bubba gump")
        self.assertEqual(TextObfuscator.decode_text(ctxt, key=key), ptxt)
        self.assertNotEqual(TextObfuscator.decode_text(ctxt), ptxt)


class TestsNumericObfuscator(unittest.TestCase):
    def test_basic_random(self):
        pnum = 12344321
        for _ in range(3):  # (results vary randomly: retry if test fails)
            try:
                obf = NumericObfuscator()
                cnum = sum(obf.encode(pnum))
                obf = NumericObfuscator()
                cnum2 = sum(obf.encode(pnum))
                self.assertNotEqual(cnum, cnum2)
                cnum = sum(obf.encode(pnum))
                self.assertNotEqual(cnum, cnum2)
                self.assertEqual(obf.decode(cnum), pnum)
                self.assertEqual(obf.decode(cnum2), pnum)
                break
            except AssertionError:
                continue
        else:
            self.assertFalse("All retries of test_basic_random failed")

    def test_basic_seeded(self):
        pnum = 12344321
        obf = NumericObfuscator(seed=314159)
        cnum = sum(obf.encode(pnum))
        obf = NumericObfuscator(seed=314159)
        cnum2 = sum(obf.encode(pnum))
        self.assertEqual(cnum, cnum2)
        cnum = sum(obf.encode(pnum))
        self.assertNotEqual(cnum, cnum2)
        self.assertEqual(obf.decode(cnum), pnum)
        self.assertEqual(obf.decode(cnum2), pnum)

    def test_meta(self):
        pnum = 12344321
        obf = NumericObfuscator()
        for _ in range(3):  # (results vary randomly: retry if test fails)
            val, meta = obf.encode(pnum)
            try:
                self.assertEqual(obf.decode(val, meta=meta), pnum)
                self.assertIsNone(obf.decode(val))
                self.assertNotEqual(obf.decode(val, check=False), pnum)
                break
            except AssertionError:
                continue
        else:
            self.assertFalse("All retries of test_meta failed")

    def test_check(self):
        pnum = 12344321
        obf = NumericObfuscator()
        cnum = sum(obf.encode(pnum)) + 1
        self.assertIsNone(obf.decode(cnum))
        self.assertNotEqual(obf.decode(cnum, check=False), pnum)

    def test_width(self):
        pnum = 12344321
        obf = NumericObfuscator()
        cnum = sum(obf.encode(pnum, width=5))
        self.assertEqual(cnum, -1)
        cnum = sum(obf.encode(pnum, width=11))
        self.assertIsNone(obf.decode(cnum), )
        self.assertEqual(obf.decode(cnum, width=11), pnum)

    def test_pail(self):
        pnum = 12344321
        obf = NumericObfuscator()
        cnum = sum(obf.encode(pnum, pail=7))
        cnum2 = sum(obf.encode(pnum, pail=7))
        self.assertEqual(cnum, cnum2)
        self.assertEqual(obf.decode(cnum), pnum)
        cnum = sum(obf.encode(pnum, pail=7))
        cnum2 = sum(obf.encode(pnum, pail=8))
        self.assertNotEqual(cnum, cnum2)
        self.assertEqual(obf.decode(cnum), pnum)
        self.assertEqual(obf.decode(cnum2), pnum)
