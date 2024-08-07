# Testing code

import numpy as np
import unittest
import subprocess
from .. import netcdf_read_write


class Tests(unittest.TestCase):

    def test_pixel_node_writer(self):
        """
        See if the writing function for pixel-node files produces a pixel-node file.
        The behavior has been finicky for float32 vs float64
        Writing a full test for float32 would be good (although the example grd file gets pretty close)
        """
        grid_def = [-120, -114, 32, 37]
        inc = [0.02, 0.02]
        filename = 'test_outfile.nc'
        lons = np.arange(grid_def[0], grid_def[1] + 0.00001, inc[0])
        lats = np.arange(grid_def[2], grid_def[3] + 0.00001, inc[1])

        # Test a write function
        grid = np.zeros((len(lats), len(lons)))
        netcdf_read_write.write_netcdf4(lons, lats, grid, filename)
        netcdf_read_write.parse_pixelnode_registration(filename)
        subprocess.call(['rm', filename], shell=False)
        subprocess.call(['rm', 'gmt.history'], shell=False)

        # Test a read-write cycle on an example grid
        [x, y, z] = netcdf_read_write.read_any_grd("Tectonic_Utils/read_write/test/example_grd.grd")
        netcdf_read_write.write_netcdf4(x, y, z, "Tectonic_Utils/read_write/test/written_example.grd")
        netcdf_read_write.parse_pixelnode_registration("Tectonic_Utils/read_write/test/written_example.grd")
        subprocess.call(['rm', 'gmt.history'], shell=False)

        return


if __name__ == "__main__":
    unittest.main()
