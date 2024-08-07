# Testing code

import numpy as np
import unittest
from .. import fault_vector_functions, insar_vector_functions, xyz2llh


class Tests(unittest.TestCase):

    def test_strike(self):
        strike = fault_vector_functions.get_strike(deltax=1.0, deltay=0.0)
        self.assertEqual(strike, 90)
        strike = fault_vector_functions.get_strike(deltax=-1.0, deltay=0.0)
        self.assertEqual(strike, 270)
        strike = fault_vector_functions.get_strike(deltax=0.0, deltay=-1.0)
        self.assertEqual(strike, 180)
        angle = -160
        strike = fault_vector_functions.get_strike(deltax=np.cos(np.deg2rad(angle)), deltay=np.sin(np.deg2rad(angle)))
        self.assertEqual(strike, 250)
        return

    def test_rake(self):
        rake = fault_vector_functions.get_rake(rtlat_strike_slip=-1.0, dip_slip=0.5)
        self.assertAlmostEqual(rake, 26.5650511770779)
        return

    def test_plane_normal(self):
        plane_normal = fault_vector_functions.get_plane_normal(strike=0, dip=0)
        self.assertTrue(np.allclose(plane_normal, [0, 0, 1]))  # the calculated answer
        plane_normal = fault_vector_functions.get_plane_normal(strike=90, dip=89.99)
        self.assertTrue(np.allclose(plane_normal, [0, -1, 0.0001745329]))  # the calculated answer
        plane_normal = fault_vector_functions.get_plane_normal(strike=270, dip=89.99)
        self.assertTrue(np.allclose(plane_normal, [0, 1, 0.0001745329]))
        plane_normal = fault_vector_functions.get_plane_normal(strike=180, dip=1)
        self.assertTrue(np.allclose(plane_normal, [-0.0174524064372835, 0, 0.999847695156391]))
        return

    def test_lkv_conversion(self):
        """ Testing that look vectors are converted correctly"""
        [lkv_e, lkv_n, lkv_u] = insar_vector_functions.flight_incidence_angles2look_vector(190, 30)
        [out_flight, out_inc] = insar_vector_functions.look_vector2flight_incidence_angles(lkv_e, lkv_n, lkv_u)
        self.assertAlmostEqual(190, out_flight)
        self.assertAlmostEqual(30, out_inc)
        return

    def test_llh_xyz_conversion(self):
        """ Testing conversion between xyz and llh coordinates """
        print("Testing llh/enu/xyz conversion math")
        llh_point = np.array([[-121.98765, 37.12345, 0],
                              [-121.98765, 38.12345, 0], ])
        xyz = xyz2llh.llh2xyz(llh_point)
        llh_back = xyz2llh.xyz2llh(xyz)
        self.assertTrue(np.allclose(llh_point, llh_back, atol=1e-6))

        # Testing the xyz to llh conversion on values from a AC11 pos file from PBO GPS data.
        xyz_point = np.array([[-2571637.61354, -1586307.97196,  5599086.71297], ])
        AC11_location = np.array([[211.6682527307-360, 61.8070788863, 790.83844], ])
        llh_point = xyz2llh.xyz2llh(xyz_point)
        self.assertTrue(np.allclose(llh_point, AC11_location))
        return

    def test_enu_conversion(self):
        # Testing enu local conversion for station AC11 PBO GPS data
        # Functions can use LLH or XYZ origin
        reference_xyz = np.array([-2571637.71108, -1586307.90018,  5599086.75442])  # 1d array from pos file
        reference_llh = np.array([211.6682506012 - 360, 61.8070787035], )            # llh reference position
        xyz_obs_2005 = np.array([[-2571637.61354, -1586307.97196,  5599086.71297], ])  # xyz position on 20050721
        enu_from_pos_file = np.array([[0.11200,  0.02035,  -0.05795], ])               # enu position on 20050721
        e_enu, _ = xyz2llh.xyz2enu(np.subtract(xyz_obs_2005, reference_xyz), reference_llh, dcov=None)
        self.assertTrue(np.allclose(e_enu, enu_from_pos_file, atol=1e-3))

        # Trying the enu to xyz coordinates
        e_xyz, _ = xyz2llh.enu2xyz(enu_from_pos_file, reference_xyz, dcov=None)
        self.assertTrue(np.allclose(np.add(e_xyz, reference_xyz), xyz_obs_2005, atol=1e-3))
        # Feel-good code: check out the print statements!
        print("pos file enu vs computed enu: ", enu_from_pos_file, e_enu)
        print("pos_xyz_obs vs computed xyz_obs: ", xyz_obs_2005, np.add(e_xyz, reference_xyz))
        return

    def test_proj_into_vertical_no_horizontal(self):
        # LKV 45 degree test case:
        semivert = insar_vector_functions.proj_los_into_vertical_no_horiz(1, [0.7, 0, 0.7])
        self.assertAlmostEqual(semivert, 1.41421356)
        # LKV vertical test case:
        semivert = insar_vector_functions.proj_los_into_vertical_no_horiz(1, [0, 0, 1])
        self.assertAlmostEqual(semivert, 1.0)
        return


if __name__ == "__main__":
    unittest.main()
