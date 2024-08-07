'''

Written by kfhe, at 12/19/2022
'''

from csi.TriangularTents import TriangularTents
from csi.TriangularPatches import TriangularPatches
from csi.gps import gps
from csi.insar import insar
import pandas as pd
import copy
import numpy as np
import pyproj as pp
import sys
import h5py
# import multifaultsolve_kfh as multfaultsolve
# from multifaultsolve_kfh import multifaultsolve_kfh as msolve


# 定义Pylith生成的格林函数相关数据
grnfns = {'ss_impluse': 'slip_ss-fault-slab.h5',
            'ds_impluse': 'slip_ds-fault-slab.h5',
            'ss_response': 'slip_ss-cgps_sites.h5',
            'ds_response': 'slip_ds-cgps_sites.h5'
            }

# --------------------------------------------------------------------------
def matchCoords(coordsRef, coords):
    """
    Function to provide indices that match the given set of coordinates to a
    reference set.
    """

    diff = coordsRef[:, :, None] - coords[:, :, None].transpose()
    dist = np.linalg.norm(diff, axis=1)
    inds = np.argmin(dist, axis=0)
    distances = dist[inds].diagonal()
    return (distances, inds)
# --------------------------------------------------------------------------


class TriangularTents_kfh(TriangularTents):
    '''
    Classes implementing a fault made of triangular tents. Inherits from Fault

    Args:
        * name      : Name of the fault.

    Kwargs:
        * utmzone   : UTM zone  (optional, default=None)
        * lon0      : Longitude of the center of the UTM zone
        * lat0      : Latitude of the center of the UTM zone
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : Speak to me (default=True)
    '''

    # ----------------------------------------------------------------------    
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, 
                        lat0=None, verbose=True):

        # Base class init
        super(TriangularTents_kfh, self).__init__(name, utmzone=utmzone, ellps=ellps, 
                                                lon0=lon0, lat0=lat0, 
                                                verbose=verbose)

        # Specify the type of patch
        self.patchType = 'triangletent'
        self.area = None
        self.area_tent = None

        # All done
        return
    # ----------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def extractfromPylith(self, grnfnsdict=None, datatype='gps', pylithproj='tmerc', 
                        faultname ='grnimpluse', dataname='grnresponse'):
        '''
        Function to read impulse and response info from PyLith output files.
        '''
        assert grnfnsdict.__class__ is dict, 'Expect a GrnFns information dictionary from Pylith'
        
        utmzone, lon0, lat0, ellps = self.utmzone, self.lon0, self.lat0, self.ellps
        data = gps(name=dataname, utmzone=utmzone, lon0=lon0, lat0=lat0, ellps=ellps)
        
        # 定义断层类型，用于后面存储数据节点集和Faces集
        trifault = TriangularPatches(name=faultname, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        print("Reading Green's functions:")
        sys.stdout.flush()

        # Read impulses.
        print("  Reading left-lateral impulses:")
        sys.stdout.flush()
        with h5py.File(grnfnsdict['ss_impluse'], 'r') as impulsesLl:
            Vertices = impulsesLl['geometry/vertices'][:]
            Vertices[:, 2] *= -1
            Faces = np.array(impulsesLl['topology/cells'][:], dtype=np.int)
            trifault.Faces = Faces
            llSlip = impulsesLl['vertex_fields/slip'][:,:,0]
            llTentsId = np.nonzero(llSlip != 0.0)[1]
            # 这里是提取除埋深边界点以外的其它点
            Tents = Vertices[llTentsId]
            # 注意：这里是重点，目的保存脉冲点顺序
            _, impulseInds = matchCoords(Vertices, Tents)
        
        # 将Pylith中的Tmerc投影改为UTM投影
        ptmerc = pp.Proj('+proj={0} +lat_0={1:.4f} +lon_0={2:.4f} +ellps=WGS84'.format(pylithproj, lat0, lon0))
        lon, lat = ptmerc(Vertices[:, 0], Vertices[:, 1], inverse=True)
        x, y = trifault.putm(lon, lat)
        trifault.Vertices = np.vstack((x, y, Vertices[:, 2])).T/1000.
        trifault.vertices2ll()
        self.initializeFromFault(trifault)
        self.vertices2tents()
        self.chooseTents(impulseInds)
        # Set Patch
        patch = self.Vertices[self.Faces, :]
        self.patch = patch
        self.patch2ll()
        self.setdepth()

        print("  Reading updip impulses:")
        sys.stdout.flush()
        with h5py.File(grnfnsdict['ds_impluse'], 'r') as impulsesUd:
            udCoords = impulsesUd['geometry/vertices'][:]
            udCoords[:, 2] *= -1.
            udSlip = impulsesUd['vertex_fields/slip'][:,:,1]
            udImpInds = np.nonzero(udSlip != 0.0)[1]
            udCoordsUsed = udCoords[udImpInds]
            _, udCoordInds = matchCoords(udCoordsUsed, Tents)

        # Read responses.
        print("  Reading left-lateral responses:")
        sys.stdout.flush()
        with h5py.File(grnfnsdict['ss_response'], 'r') as responseLl:
            llResponseCoords = responseLl['geometry/vertices'][:]
            llResponseVals = responseLl['vertex_fields/displacement'][:]
            sta = responseLl['stations'][:].astype(dtype='U4')
            
            llResponsesEast  = llResponseVals[:, :, 0]
            llResponsesNorth = llResponseVals[:, :, 1]
            llResponsesUp    = llResponseVals[:, :, 2]

        stalon, stalat = ptmerc(llResponseCoords[:, 0], llResponseCoords[:, 1], inverse=True)
        data.setStat(sta, stalon, stalat, loc_format='LL', initVel=True)

        print("  Reading updip responses:")
        sys.stdout.flush()
        with h5py.File(grnfnsdict['ds_response'], 'r') as responseUd:
            responseUdVals = responseUd['vertex_fields/displacement'][:]
            # 这里使得逆冲脉冲和左旋脉冲的节点顺序对齐
            udResponseVals = responseUdVals[udCoordInds,:,:]

            udResponsesEast  = udResponseVals[:, :, 0]
            udResponsesNorth = udResponseVals[:, :, 1]
            udResponsesUp    = udResponseVals[:, :, 2]

        # Create design matrix.
        print("  Creating design matrix:")
        sys.stdout.flush()
        self.G = {}
        self.G[dataname] = {}
        G = self.G[dataname]
        Green_ss = np.vstack((llResponsesEast.T, llResponsesNorth.T, llResponsesUp.T))
        Green_ds = np.vstack((udResponsesEast.T, udResponsesNorth.T, udResponsesUp.T))
        G['strikeslip'] = Green_ss
        G['dipslip'] = Green_ds
        self.grndata = data

        return self, data
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def matchGrnresponse2Stas(self, data, los=None, vertical=True, match='name'):
        '''
        match: 'coord' or 'name'
        data: GPS or InSAR
        '''
        assert hasattr(self, 'grndata'), 'There is not a greenfunctions from pylith'
        if match == 'name':
            res_grn_sta = self.grndata.station
            datagrnidx = [np.where(res_grn_sta == sta)[0].item(0) for sta in data.station]
        elif match == 'coord':
            # 目的：因为每次输出站点点位顺序均不同，这里根据坐标判断
            res_grn_coord = np.vstack((self.grndata.x, self.grndata.y)).T
            _, datagrnidx = matchCoords(res_grn_coord, np.vstack((data.x, data.y)).T)

        Gss = self.G['grnresponse']['strikeslip']
        Gds = self.G['grnresponse']['dipslip']
        nd = int(Gss.shape[0]/3)
        ncomp = 3 if vertical else 2
        dataname = data.name
        if dataname not in self.G:
            self.G[dataname] = {}
        G = self.G[dataname]
        
        if los is None:
            G_ss, G_ds = [], []
            for i in range(ncomp):
                G_ss.append(Gss[np.array(datagrnidx)+i*nd, :])
                G_ds.append(Gds[np.array(datagrnidx)+i*nd, :])
            G_ss = np.vstack(G_ss)
            G_ds = np.vstack(G_ds)
        elif los.shape[0] == data.vel.shape[0]:
            G_ss, G_ds = 0, 0
            for i in range(ncomp):
                G_ss += Gss[np.array(datagrnidx)+i*nd, :]*data.los[:, i][:, None]
                G_ds += Gds[np.array(datagrnidx)+i*nd, :]*data.los[:, i][:, None]
        G['strikeslip'] = G_ss
        G['dipslip'] = G_ds

        return G
    # --------------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildAdjacencyMap(self, verbose=True):
        '''
        For each triangle vertex, find the indices of the adjacent triangles.
        This function overwrites that from the parent class TriangularPatches.

        Kwargs:
            * verbose       : Speak to me

        Returns:
            * None
        '''

        if verbose:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Finding the adjacent triangles for all vertices")

        self.adjacencyMap = []

        # Cache the vertices and faces arrays
        tentid, faces = self.tentid, np.array(self.Faces)

        # First find adjacent triangles for all triangles
        numvert = len(tentid)
        numface = len(faces)

        for i in range(numvert):
            if verbose:
                sys.stdout.write('%i / %i\r' % (i + 1, numvert))
                sys.stdout.flush()

            # Find triangles that share an edge
            adjacents = []
            for j in range(numface):
                if tentid[i] in faces[j,:]:
                    adjacents.append(j)

            self.adjacencyMap.append(adjacents)

        if verbose:
            print('\n')
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildTentAdjacencyMap(self, verbose=True):
        '''
        For each triangle vertex, finds the indices of the surrounding vertices.
        This function runs typically after buildAdjacencyMap.

        Kwargs:
            * verbose       : Speak to me

        Returns:
            * None
        '''

        if verbose:
            print("-----------------------------------------------")
            print("-----------------------------------------------")
            print("Finding the adjacent vertices for all vertices.")

        # Check adjacency Map
        if not hasattr(self, 'adjacencyMap'):
            self.buildAdjacencyMap(verbose=verbose)

        # Cache adjacencyMap
        adjacency = self.adjacencyMap 
        faces = self.Faces

        # Create empty lists
        adjacentTents = []

        # Iterate over adjacency map
        for adj, iVert in zip(adjacency, self.tentid):
            # Create a list for that tent
            tent = []
            # Iterate over the surrounding triangles
            for iTriangle in adj:
                face = faces[iTriangle]
                face = face[face!=iVert]
                tent.append(face)
            # Clean up tent
            tent = np.unique(np.concatenate(tent)).tolist()
            # Append
            adjacentTents.append(tent)

        # Save
        self.adjacentTents = adjacentTents

        # All don
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian(self, verbose=True, method='distance', irregular=False):
        '''
        Build a discrete Laplacian smoothing matrix.

        Args:
            * verbose       : if True, displays stuff.
            * method        : Method to estimate the Laplacian operator

                - 'count'   : The diagonal is 2-times the number of surrounding nodes. Off diagonals are -2/(number of surrounding nodes) for the surrounding nodes, 0 otherwise.
                - 'distance': Computes the scale-dependent operator based on Desbrun et al 1999. (Mathieu Desbrun, Mark Meyer, Peter Schr\"oder, and Alan Barr, 1999. Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow, Proceedings of SIGGRAPH).

            * irregular     : Not used, here for consistency purposes

        Returns:
            * Laplacian     : 2D array
        '''
 
        # Build the tent adjacency map
        self.buildTentAdjacencyMap(verbose=verbose)

        # Get the vertices
        vertices = self.Vertices

        # Allocate an array
        D = np.zeros((len(self.tentid), len(self.tentid)))

        # Normalize the distances
        if method=='distance':
            self.Distances = []
            for adja, i in zip(self.adjacentTents, self.tentid):
                x0, y0, z0 = vertices[i,0], vertices[i,1], vertices[i,2]
                xv, yv, zv = vertices[adja,0], vertices[adja,1], vertices[adja,2] 
                distances = np.array([np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2) 
                    for x, y, z in zip(xv, yv, zv)])
                self.Distances.append(distances)
            normalizer = np.max([d.max() for d in self.Distances])

        # Iterate over the vertices
        i = 0
        for adja in self.adjacentTents:
            adjtent = list(set(adja) & set(self.tentid))
            adjind = [np.argwhere(np.array(self.tentid) == id).item(0) for id in adjtent]
            # Counting Laplacian
            if method=='count':
                D[i,i] = 2*float(len(adja))
                D[i,adjind] = -2./float(len(adja))
            # Distance-based
            elif method=='distance':
                ind = [np.argwhere(np.array(adja) == adj).item(0) for adj in adjtent]
                distances = self.Distances[i]/normalizer
                E = np.sum(distances) # [ind]
                D[i,i] = 2./E * np.sum(1./distances) # float(len(adja))* [ind]
                D[i,adjind] = -2./E * 1./distances[ind]

            # Increment 
            i += 1

        # All done
        return D
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian_kfh(self, verbose=True, method='distance', irregular=False):
        """
        Get the smoothing matrix ,by cells verteix adjacent num & distance
        self.faultCells & self.faultVertCoords & self.impulseInds
        """
        size = len(self.tentid)
        regArray = np.zeros((size, size), dtype=np.float)
        # get the distance pnt to pnt
        vertices = np.array(self.Vertices)
        diff = vertices[:, :, None] - vertices[:, :, None].transpose()
        dist = np.linalg.norm(diff, axis=1)
        # 
        for i in range(size):
            ind = self.tentid[i]
            rows, _ = np.where(self.Faces==ind)
            inds = set(self.Faces[rows, :].flatten().tolist())
            inds.remove(ind)
            cohij = inds & set(self.tentid)
            hij = dist[ind, list(inds)]
            # hij = dist[ind, list(cohij)]
            #print(hij)
            L = np.sum(hij)
            inv_hijsum = np.sum(1.0/hij)
            regArray[i, i] = -2.0/L*inv_hijsum
            for index in cohij:
                row,  = np.where(np.array(self.tentid)==index)
                regArray[i, row[0]] = 2.0/(L*dist[ind][index])
        return regArray
