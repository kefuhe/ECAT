import unittest
from .. import second_focal_plane


class Tests(unittest.TestCase):

    def test_simple_focal_plane(self):
        print("Testing second focal plane.")
        np1 = [38, 51, 44]
        expected_outcome = [276.71179351444425, 57.32650724684216, 131.61502732561564]
        np2 = second_focal_plane.find_aux_plane(np1[0], np1[1], np1[2])
        self.assertEqual(np2, expected_outcome, 'Wrong focal plane')


if __name__ == '__main__':
    unittest.main()
