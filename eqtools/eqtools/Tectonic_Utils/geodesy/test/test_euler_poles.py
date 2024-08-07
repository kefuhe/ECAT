# Testing code

import unittest
import numpy as np
from .. import euler_pole


def flush_euler_vector_to_euler_pole(euler_pole_start):
    """ Convert an euler pole to an euler vector and back. """
    euler_vector = euler_pole.euler_pole_to_euler_vector(euler_pole_start)
    euler_pole_returned = euler_pole.euler_vector_to_euler_pole(euler_vector)
    return euler_pole_returned


class Tests(unittest.TestCase):

    def test_euler_vector_and_pole(self):
        """Test the math converting euler poles and euler vectors"""
        euler_pole_start = [-120, 40, 0.22]
        euler_pole_returned = flush_euler_vector_to_euler_pole(euler_pole_start)
        self.assertTrue(np.allclose(list(euler_pole_start), list(euler_pole_returned)))
        euler_pole_start = [-120, 40, -0.22]
        euler_pole_returned = flush_euler_vector_to_euler_pole(euler_pole_start)
        euler_pole_returned = euler_pole.get_opposite_ep(euler_pole_returned)
        self.assertTrue(np.allclose(list(euler_pole_start), list(euler_pole_returned)))
        euler_pole_start = [-120, -40, -0.22]
        euler_pole_returned = flush_euler_vector_to_euler_pole(euler_pole_start)
        euler_pole_returned = euler_pole.get_opposite_ep(euler_pole_returned)
        self.assertTrue(np.allclose(list(euler_pole_start), list(euler_pole_returned)))


if __name__ == "__main__":
    unittest.main()
