# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import matplotlib.path as path
import matplotlib as mpl
import copy
import sys
import os
from scipy.interpolate import griddata
import multiprocessing as mp
from scipy.spatial import KDTree
import matplotlib.path as mpath
import gmsh
from scipy.spatial import Delaunay
from scipy.linalg import block_diag
from scipy.optimize import least_squares

# Personals
from .insar import insar
from .opticorr import opticorr
from .imagecovariance import imagecovariance as imcov
from .csiutils import _split_seq
from .imagedownsampling import imagedownsampling


class imagedownsamplingTriangular(imagedownsampling):
    def __init__(self, name, image, faults=None, verbose=True, vel_type='east'):
        '''
        Initialize the downsampler.

        Args:
            * name       : Name of the downsampler.
            * image      : Image data.
            * faults     : Faults data.

        Kwargs:
            * verbose    : Whether to print verbose output. Default is True.
            * vel_type   : Type of velocity to use ('east', 'north', or 'magnitude'). Default is 'east'.
        '''
        # Initialize the downsampler
        super(imagedownsamplingTriangular, self).__init__(name, image, faults=faults, verbose=verbose)
        self.vel_type = vel_type  # Store the velocity type

        # Adapt velocity based on vel_type
        if self.datatype == 'opticorr':
            if self.vel_type == 'east':
                self.image.vel = self.image.east
                self.image.err = self.image.err_east
            elif self.vel_type == 'north':
                self.image.vel = self.image.north
                self.image.err = self.image.err_north
            elif self.vel_type == 'eastnorth':
                self.image.vel = np.hstack((self.image.east, self.image.north))
                self.image.err = np.hstack((self.image.err_east, self.image.err_north))
            else:
                raise ValueError(f"Invalid vel_type: {self.vel_type}. Must be 'east', 'north', or 'eastnorth'.")

    def initialstate(self, minimumsize, tolerance=0.25, plot=False, decimorig=10, boundary_tolerance=0.5):
        '''
        Generates an initial triangular mesh using Delaunay triangulation and adds perturbations.

        Args:
            * minimumsize      : Minimum size of the blocks in the mesh.

        Kwargs:
            * tolerance         : Between 0 and 1. If 1, all the pixels must have a value so that the box is kept. If 0, no pixels are needed... Default is 0.25
            * plot              : True/False
            * decimorig         : Decimation for plotting purposes only.
            * boundary_tolerance: Tolerance for setting points to boundary values. Default is 0.5.

        Returns:
            * None
        '''
        self.tolerance = tolerance
        self.minsize = minimumsize

        # Define the grid points
        x = np.linspace(self.xmin, self.xmax, 10)
        y = np.linspace(self.ymin, self.ymax, 10)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()

        # Add random perturbations
        x += np.random.randn(x.size)
        y += np.random.randn(y.size)

        # Ensure points are within the boundaries with boundary tolerance
        x = np.where(np.abs(x - self.xmin) < boundary_tolerance, self.xmin, x)
        x = np.where(np.abs(x - self.xmax) < boundary_tolerance, self.xmax, x)
        y = np.where(np.abs(y - self.ymin) < boundary_tolerance, self.ymin, y)
        y = np.where(np.abs(y - self.ymax) < boundary_tolerance, self.ymax, y)

        # Combine x and y into a single array of points
        points = np.vstack((x, y)).T

        # Perform Delaunay triangulation
        delaunay = Delaunay(points)
        mesh_points = delaunay.points
        mesh_elements = delaunay.simplices

        # Store the mesh in the object
        self.mesh_points = mesh_points
        self.mesh_elements = mesh_elements

        # Optionally plot the mesh
        if plot:
            plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements)
            plt.scatter(x, y, color='red', marker='o')
            plt.title('Initial Triangular Mesh with Perturbations')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        # Set the triangular blocks
        self.setBlocks(mesh_points, mesh_elements)

        # Generate the sampling to test
        self.downsample(plot=False, decimorig=decimorig)

        # All done
        return
    
    def setBlocks(self, mesh_points, mesh_elements):
        '''
        Takes mesh points and elements and sets them in self.
    
        Args:
            * mesh_points    : Array of mesh points (coordinates)
            * mesh_elements  : Array of mesh elements (triangles)
    
        Returns:
            * None
        '''
    
        # Save the mesh points and elements
        self.mesh_points = mesh_points
        self.mesh_elements = mesh_elements
    
        # Initialize lists for triangles in xy and lon, lat coordinates
        triangles = []
        trianglesll = []
    
        # Build the list of triangles in both xy and lon, lat coordinates
        for element in mesh_elements:
            p1, p2, p3 = element
            tri_xy = [[mesh_points[p1][0], mesh_points[p1][1]],
                      [mesh_points[p2][0], mesh_points[p2][1]],
                      [mesh_points[p3][0], mesh_points[p3][1]]]
            tri_ll = [self.xy2ll(mesh_points[p1][0], mesh_points[p1][1]),
                      self.xy2ll(mesh_points[p2][0], mesh_points[p2][1]),
                      self.xy2ll(mesh_points[p3][0], mesh_points[p3][1])]
            triangles.append(tri_xy)
            trianglesll.append(tri_ll)
    
        # Save the triangles
        self.triangles = triangles
        self.trianglesll = trianglesll
    
        # Set blocks
        self.blocks = self.triangles
        self.blocksll = self.trianglesll
    
        # All done
        return
    
    def getblockcenter(self, block):
        """
        Calculate the centroid of a triangular block.
    
        Args:
            block (list of tuples): A list of three tuples, each representing the (x, y) coordinates of a vertex of the triangle.
    
        Returns:
            tuple: The (x, y) coordinates of the centroid of the triangle.
        """
        t1, t2, t3 = block
        x1, y1 = t1
        x2, y2 = t2
        x3, y3 = t3
    
        xc = (x1 + x2 + x3) / 3.0
        yc = (y1 + y2 + y3) / 3.0
        return xc, yc
    
    def getblockarea(self, block):
        """
        Calculate the area of a triangular block.
    
        Args:
            block (list of tuples): A list of three tuples, each representing the (x, y) coordinates of a vertex of the triangle.
    
        Returns:
            float: The area of the triangle.
        """
        t1, t2, t3 = block
        x1, y1 = t1
        x2, y2 = t2
        x3, y3 = t3
    
        # Using the Shoelace formula to calculate the area of the triangle
        area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        return area

    def cutblocksbygmsh(self, X, Y, smoothwidths, show_mesh=True, verbose=0):
        """
        Generate a mesh using Gmsh and control mesh size based on given smooth widths.
    
        Parameters:
        ----------
        X, Y : np.ndarray
            x and y coordinates of the mesh points.
        smoothwidths : np.ndarray
            Smooth widths controlling the mesh size.
        show_mesh : bool, optional
            Whether to display the generated mesh, default is True.
        verbose : int, optional
            Verbosity level for Gmsh printing information, default is 0.
        """
        from scipy.interpolate import griddata
    
        # Define the smooth width function
        def hfun1(x, y, smoothwidths, X, Y):
            points = np.column_stack((X, Y))
            values = smoothwidths
            h = griddata(points, values, (x, y), method='linear')
            h_nearest = griddata(points, values, (x, y), method='nearest')
            if np.isnan(h):
                h = h_nearest
            return h
    
        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", verbose)
        gmsh.model.add("smooth_mesh")
    
        # Define boundary nodes
        points = []
        for inodell in self.boxll:
            ilon, ilat = inodell
            ix, iy = self.ll2xy(ilon, ilat)
            points.append(gmsh.model.geo.addPoint(ix, iy, 0, 0.1))
    
        # Create lines and closed curve
        lines = [
            gmsh.model.geo.addLine(points[0], points[1]),
            gmsh.model.geo.addLine(points[1], points[2]),
            gmsh.model.geo.addLine(points[2], points[3]),
            gmsh.model.geo.addLine(points[3], points[0])
        ]
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
    
        # Set mesh size function
        def mesh_size_callback(dim, tag, x, y, z, lc):
            h = hfun1(x, y, smoothwidths, X, Y)
            return float(h)
    
        gmsh.model.mesh.setSizeCallback(mesh_size_callback)
    
        # Generate geometry and mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
    
        # gmsh.fltk.run()
        # Extract mesh points and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        _, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
        mesh_points = np.array(node_coords).reshape(-1, 3)[:, :2]
        mesh_elements = np.array(node_tags[0]).reshape(-1, 3) - 1
    
        # Store the mesh in the object
        self.mesh_points = mesh_points
        self.mesh_elements = mesh_elements
    
        # Optionally plot the mesh
        if show_mesh:
            import matplotlib.pyplot as plt
            plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements)
            plt.scatter(X, Y, color='red', marker='o')
            plt.title('Generated Triangular Mesh with Smooth Widths')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
    
        # Finalize gmsh
        gmsh.finalize()
    
        # Set the triangular blocks
        self.setBlocks(mesh_points, mesh_elements)
    
        # All done
        return

    def resolutionBased(self, max_samples=5000, change_threshold=5, smooth_factor=0.25, slipdirection='s', plot=False, 
                        verboseLevel='minimum', decimorig=10, vertical=False):
        '''
        Iteratively downsamples the dataset until the number of samples is below max_samples and the relative change in the number of blocks is less than change_threshold.
    
        Args:
            * max_samples     : Maximum number of samples. Default is 5000.
            * change_threshold: Relative change threshold for the number of blocks. Default is 5%.
            * smooth_factor   : Smoothing factor for the resolution matrix. Default is 0.25.
    
        Kwargs:
            * slipdirection   : Which direction to account for to build the slip Green's functions (s, d or t)
            * plot            : False/True
            * verboseLevel    : talk to me
            * decimorig       : Decimate a bit before plotting
            * vertical        : Use vertical green's functions.
    
        Returns:
            * None
        '''
    
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Downsampling Iterations")
    
        # Check if vertical is set properly
        if not vertical and self.datatype == 'insar':
            print("----------------------------------")
            print("----------------------------------")
            print(" Watch Out!!!!")
            print(" We have set vertical to True, because ")
            print(" LOS is always very sensitive to vertical")
            print(" displacements...")
            vertical = True
    
        ns = len(slipdirection)
        # Laplacian smoothing
        smooth = None
        for fault in self.faults:
            D = fault.buildLaplacian(method='Mudpy', bounds=['free',]*4, 
                                       topscale=0.1, bottomscale=0.1)
            if ns == 2:
                D = block_diag(D, D)
            
            if smooth is None:
                smooth = D
            else:
                smooth = block_diag(smooth, D)
        
        self.smooth = smooth
        self.smooth /= -4.0
        self.Var = np.var(self.image.vel[np.isfinite(self.image.vel)])
    
        # Creates the variable that is supposed to stop the loop
        do_cut = False
    
        # counter
        it = 0
    
        # Check if block size is minimum
        Bsize = self._is_minimum_size(self.blocks)
    
        # Initialize the number of samples
        old_num_samples = 0
        done = False
    
        # Loops until done
        while not done:
            # Cut if asked
            if do_cut:
                self.cutblocksbygmsh(self.newimage.x, self.newimage.y, self.smoothwidths, show_mesh=False)
                # Set the blocks
                self.setBlocks(self.mesh_points, self.mesh_elements)
                # Do the downsampling
                self.downsample(plot=False, decimorig=decimorig)
            else:
                do_cut = True
    
            # Iteration #
            it += 1
            if self.verbose:
                print('Iteration {}: Testing {} data samples '.format(it, len(self.blocks)))
    
            # Compute resolution
            self.computeSmoothwidth(slipdirection, smooth_factor, vertical=vertical)
    
            # Blocks that have a minimum size, don't check these
            Bsize = self._is_minimum_size(self.blocks)
            self.smoothwidths[np.where(Bsize)] = self.minsize
    
            # Calculate the number of samples and the relative change
            num_samples = len(self.blocks)
            perc_change = abs(old_num_samples - num_samples) / num_samples * 100
    
            if self.verbose and verboseLevel != 'minimum':
                sys.stdout.write(' ===> Resolution from {} to {}, Mean = {} +- {} \n'.format(self.smoothwidths.min(),
                    self.smoothwidths.max(), self.smoothwidths.mean(), self.smoothwidths.std()))
                sys.stdout.flush()
    
            # Plot at the end of that iteration
            if plot:
                self.plotDownsampled(decimorig=decimorig)
    
            # Check stopping criteria
            if num_samples > max_samples or perc_change < change_threshold:
                print('Stopping criteria met: {} samples and {}% change'.format(num_samples, perc_change))
                done = True
    
            # Update the number of samples
            old_num_samples = num_samples
    
        if self.verbose:
            print(" ")
    
        # All done
        return

    def computeResolutionMatrix(self, slipdirection, smooth_factor=0.25, vertical=False):
        '''
        Computes the resolution matrix in the data space.
    
        Args:
            * slipdirection : Directions to include when computing the resolution operator.
            * smooth_factor : Smoothing factor for the resolution matrix. Default is 0.25.
    
        Kwargs:
            * vertical      : Use vertical GFs?
    
        Returns:
            * None
        '''
    
        # Check if vertical is set properly
        if not vertical and self.datatype == 'insar':
            print("----------------------------------")
            print("----------------------------------")
            print(" Watch Out!!!!")
            print(" We have set vertical to True, because ")
            print(" LOS is always very sensitive to vertical")
            print(" displacements...")
            vertical = True
    
        # Create the Greens function
        G = None
    
        # Compute the greens functions for each fault and concatenate these together
        for fault in self.faults:
            # build GFs
            fault.buildGFs(self.newimage, vertical=vertical, slipdir=slipdirection, verbose=False)
            fault.assembleGFs([self.newimage], polys=None, slipdir=slipdirection, verbose=False)
            if self.datatype == 'opticorr':
                if self.vel_type == 'eastnorth':
                    pass
                elif self.vel_type == 'east':
                    fault.Gassembled = fault.Gassembled[:len(self.newimage.east), :]
                elif self.vel_type == 'north':
                    fault.Gassembled = fault.Gassembled[len(self.newimage.east):, :]
            # Concatenate GFs
            if G is None:
                G = fault.Gassembled
            else:
                G = np.hstack((G, fault.Gassembled))
    
        # Compute the center points of the blocks
        centers = np.array([self.getblockcenter(block) for block in self.blocks])
        x = (centers[:, 0] - np.min(centers[:, 0])) / (np.max(centers[:, 0]) - np.min(centers[:, 0]))
        y = (centers[:, 1] - np.min(centers[:, 1])) / (np.max(centers[:, 1]) - np.min(centers[:, 1]))
    
        # Diagonal matrix Cd
        # Var = np.std(self.image.vel) * np.std(self.image.vel)
        if self.datatype == 'opticorr':
            if self.vel_type == 'eastnorth':
                Var = np.std(self.image.east) * np.std(self.image.north)
                npaches = len(self.newimage.east) + len(self.newimage.north)
                Cd = np.diag(Var * np.ones(npaches) / np.hstack((self.newimage.wgt, self.newimage.wgt)))
            elif self.vel_type == 'east':
                Var = np.std(self.image.east) * np.std(self.image.east)
                npaches = len(self.newimage.east)
                Cd = np.diag(Var * np.ones(npaches) / self.newimage.wgt)
            elif self.vel_type == 'north':
                Var = np.std(self.image.north) * np.std(self.image.north)
                npaches = len(self.newimage.north)
                Cd = np.diag(Var * np.ones(npaches) / self.newimage.wgt)
        if self.datatype == 'insar':
            Var = np.std(self.image.vel) * np.std(self.image.vel)
            npaches = len(self.newimage.vel)
            Cd = np.diag(Var * np.ones(npaches) / self.newimage.wgt)
        ch = np.linalg.cholesky(Cd)
        Cdinv = np.linalg.inv(ch.T)
    
        # Ramp matrix
        Gramp = np.vstack([np.ones(npaches), x, y, x * y]).T
        nramp = 4
    
        # Combine G and Gramp
        G = np.dot(Cdinv, np.hstack([G, Gramp]))
    
        # Smoothing matrix
        smooth = smooth_factor * self.smooth
        smoothW = np.hstack([smooth, np.zeros((smooth.shape[0], nramp))])
        Gsmoo = np.vstack([G, smoothW])
    
        # Compute the mixure resolution matrix
        # Pseudo-inverse: G^{+} = (G_s^T G_s)^{-1} G^T
        Gg = np.linalg.inv(np.dot(Gsmoo.T, Gsmoo)).dot(G.T)
        N = np.dot(G, Gg)
        self.Rd = N
    
        # If we are dealing with opticorr data, the diagonal is twice as long as the number of blocks
        if self.datatype == 'opticorr' and self.vel_type == 'eastnorth':
            Ndat = int(G.shape[0] / 2)
            self.Rd = np.sqrt(self.Rd[:Ndat]**2 + self.Rd[-Ndat:]**2)
    
        # Return the resolution matrix and centers
        return self.Rd, centers
    
    def computeSmoothwidth(self, slipdirection, smooth_factor=0.25, vertical=False):
        '''
        Computes the smooth widths based on the resolution matrix.
    
        Args:
            * slipdirection : Directions to include when computing the resolution operator.
            * smooth_factor : Smoothing factor for the resolution matrix. Default is 0.25.
    
        Kwargs:
            * vertical      : Use vertical GFs?
    
        Returns:
            * None
        '''
    
        # Compute the resolution matrix
        Rd, centers = self.computeResolutionMatrix(slipdirection, smooth_factor, vertical)
    
        # Compute smooth widths
        smoothwidths = np.zeros(len(self.blocks))
        stats = np.zeros((len(self.blocks), 3))
        for i in range(len(self.blocks)):
            x1 = centers[i, 0]
            y1 = centers[i, 1]
    
            dists = np.sqrt((centers[:, 0] - x1)**2 + (centers[:, 1] - y1)**2)
            b = 0
            x2 = dists
            y2 = Rd[i, :len(self.blocks)]
    
            minx = np.min(x2[x2 > 0]) / 2
            maxx = np.max(dists) / 2
    
            xstart = np.abs(np.sum(x2 * y2) / np.sum(y2))
            if xstart < minx:
                xstart = minx
            elif xstart > 0.75 * maxx:
                xstart = 0.75 * maxx
    
            # Fit Gaussian function
            def gausfun(params, x, y):
                return params[0] * np.exp(-((x - b)**2) / params[1]**2) - y

            Rd_init = Rd[i, i]
            Rd_init = Rd_init if Rd_init < 1. else 0.9
            Rd_init = Rd_init if Rd_init > 0. else 0.1
            initial_guess = [Rd_init, xstart]
            bounds = ([0, minx], [1, maxx])
            # print(initial_guess, bounds)
            result = least_squares(gausfun, initial_guess, bounds=bounds, args=(x2, y2))
    
            smoothwidths[i] = result.x[1]
            stats[i, :] = [xstart, result.x[1], result.cost]
    
        self.smoothwidths = smoothwidths
        self.stats = stats
    
        # All done
        return

    def readDownsamplingScheme(self, prefix):
        '''
        Reads a downsampling scheme from a rsp file and set it as self.blocks
    
        Args:
            * prefix          : Prefix of a .rsp file written by writeDownsampled2File.
    
        Returns:
            * None
        '''
    
        # Replace spaces
        prefix = prefix.replace(" ", "_")
    
        # Open file
        frsp = open(prefix+'.rsp', 'r')
    
        # Create a block list
        blocks = []
    
        # Read all the file
        Lines = frsp.readlines()
    
        # Close the file
        frsp.close()
    
        # Loop
        for line in Lines[2:]:
            x1, y1, x2, y2, x3, y3 = [float(line.split()[i]) for i in range(2, 8)]
            c1 = [x1, y1]
            c2 = [x2, y2]
            c3 = [x3, y3]
            blocks.append([c1, c2, c3])
    
        # Set the blocks
        self.setBlocks(blocks)
    
        # All done
        return
    
    def writeDownsampled2File(self, prefix, rsp=False):
        '''
        Writes the downsampled image data to a file. The file will be called prefix.txt. If rsp is True, then it writes a file called prefix.rsp containing the boxes of the downsampling. If prefix has white spaces, those are replaced by "_".
    
        Args:
            * prefix        : Prefix of the output file
    
        Kwargs:
            * rsp           : Write the rsp file?
    
        Returns:
            * None
        '''
    
        # Replace spaces
        prefix = prefix.replace(" ", "_")
    
        # Open files
        ftxt = open(prefix+'.txt', 'w')
        if rsp:
            frsp = open(prefix+'.rsp', 'w')
    
        # Write the header
        if self.datatype == 'insar':
            ftxt.write('Number xind yind east north data err wgt Elos Nlos Ulos\n')
        elif self.datatype == 'opticorr':
            ftxt.write('Number Lon Lat East North EastErr NorthErr \n')
        ftxt.write('********************************************************\n')
        if rsp:
            frsp.write('xind yind Vertex1-lon,lat Vertex2-lon,lat Vertex3-lon,lat\n')
            frsp.write('********************************************************\n')
    
        # Loop over the samples
        for i in range(len(self.newimage.x)):
    
            # Write in txt
            wgt = self.newimage.wgt[i]
            x = int(self.newimage.x[i])
            y = int(self.newimage.y[i])
            lon = self.newimage.lon[i]
            lat = self.newimage.lat[i]
            if self.datatype == 'insar':
                vel = self.newimage.vel[i]
                err = self.newimage.err[i]
                elos = self.newimage.los[i,0]
                nlos = self.newimage.los[i,1]
                ulos = self.newimage.los[i,2]
                strg = '{:4d} {:4d} {:4d} {:3.6f} {:3.6f} {} {} {} {} {} {}\n'\
                    .format(i, x, y, lon, lat, vel, err, wgt, elos, nlos, ulos)
            elif self.datatype == 'opticorr':
                east = self.newimage.east[i]
                north = self.newimage.north[i]
                err_east = self.newimage.err_east[i]
                err_north = self.newimage.err_north[i]
                strg = '{:4d} {:3.6f} {:3.6f} {} {} {} {} \n'\
                        .format(i, lon, lat, east, north, err_east, err_north)
            ftxt.write(strg)
    
            # Write in rsp
            if rsp:
                x1, y1 = self.blocksll[i][0]
                x2, y2 = self.blocksll[i][1]
                x3, y3 = self.blocksll[i][2]
                strg = '{:4d} {:4d} {} {} {} {} {} {}\n'\
                        .format(x, y, x1, y1, x2, y2, x3, y3)
                frsp.write(strg)
    
        # Close the files
        ftxt.close()
        if rsp:
            frsp.close()
    
        # All done
        return
    
    def _is_minimum_size(self, blocks):
        '''
        Returns a Boolean array. True if block is minimum size, False either.
        '''
    
        # Initialize
        Bsize = []
    
        # loop
        for block in self.blocks:
            area = self.getblockarea(block)
            if np.sqrt(area) <= self.minsize:
                Bsize.append(True)
            else:
                Bsize.append(False)
    
        # All done
        return Bsize
    
    def distToFault(self, block):
        '''
        Returns distance from block to fault. The distance is here defined as the minimum distance from any of the three block corners to the fault. (R.Grandin, April 2015)
    
        Args:
            * block     : Block instance of the imagedownsampling class.
    
        Returns:
            * float     : Minimum distance from the block to the fault.
        '''
    
        # Get the three corners
        c1, c2, c3 = block
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
    
        # Faults
        distMin = float('inf')
        for fault in self.faults:
            distCorner1 = np.min(np.hypot(fault.xf - x1, fault.yf - y1))
            distCorner2 = np.min(np.hypot(fault.xf - x2, fault.yf - y2))
            distCorner3 = np.min(np.hypot(fault.xf - x3, fault.yf - y3))
            distMin = min(distMin, distCorner1, distCorner2, distCorner3)
    
        # all done
        return distMin
    
    def blockSize(self, block):
        '''
        Returns block size. R.Grandin, April 2015
    
        Args:
            * block     : Block instance of the imagedownsampling class.
    
        Returns:
            * float     : Size of the block (square root of the area of the triangle).
        '''
    
        # compute the size (area of the triangle)
        area = self.getblockarea(block)
        size = np.sqrt(area)
    
        # all done
        return size
    
    def _isItAGoodBlock(self, block, num):
        '''
        Returns True or False given the criterion
    
        Args:
            * block     : Shape of the block
            * num       : Number of pixels
        '''
    
        if self.tolerance < 1.:
            coveredarea = num * self.pixelArea
            blockarea = self.getblockarea(block)
            return coveredarea / blockarea > self.tolerance
        else:
            return num >= self.tolerance
    
        # All done
        return

#EOF
