import unittest
from .. import wells_and_coppersmith as wc


class Tests(unittest.TestCase):

    def test_simple_focal_plane(self):
        print("Testing WC")
        mag = 6.8
        RLD = wc.RLD_from_M(mag, "N")
        RW = wc.RW_from_M(mag, "N")
        expected_RLD = 33.11311214825911  # km
        expected_RW = 17.378008287493753  # km
        self.assertEqual(RLD, expected_RLD, 'Wells and Coppersmith wrong')
        self.assertEqual(RW, expected_RW, 'Wells and Coppersmith wrong')


if __name__ == '__main__':
    unittest.main()
