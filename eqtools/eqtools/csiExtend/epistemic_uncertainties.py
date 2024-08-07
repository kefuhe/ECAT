# -*- coding: utf-8 -*-
'''
An ensemble of classes dealing with uncertainties in the fault geometry
Written by T. Ragon October 2018
'''

import numpy as np  # requires numpy version > 1.10 !
from scipy.stats import linregress
import os
import pyproj as pp


class changeFault:
    """
    Modify the fault geometry, to be used to calculate Green's Functions
    """
    def __init__(self,faultdir,filename,nstrike,utmzone,inputCoordinates='lonlatz', nbrcorners=4, verbose=True):
        """
        Open and read fault file. The file need to be in GMT format:
        lon lat z for each patch corner (or x y z)
        list of patches separated by '>'
        
        :Args:
            * faultdir  : Path of the directory where to find fault file
            * filename  : name of the fault file, or list of names
            * nstrike   : number of patches along strike for the first row (the closest to the Earth surface)
            * utmzone   : utm zone of the data
            
         :Optional Args:
            * inputCoordinates  : 'lonlatz' or 'xyz'. 'lonlatz' by default
            * nbrcorners        : number of corners for each patch, by default is 4
            
        """
        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing prior fault geometry")
            
        self.savedir = faultdir       
        self.filename = filename
        self.nstrike = nstrike
        self.utmzone = utmzone
        self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps='WGS84')
        self.inputCoordinates = inputCoordinates
        self.patch = []
        self.patchll = []
        
        # Read fault file
        fgfile = open(self.savedir+self.filename, 'r') 
        L = fgfile.readlines()

        i = 0
        while i<len(L):
            p=[]
            pll=[]
            # get the values
            if inputCoordinates in ('lonlatz'):
                for j in range(1,nbrcorners+1):
                    lon, lat, z = L[i+j].split()
                    # Pass as floating point
                    lon = float(lon); lat = float(lat); z = float(z)
                    # translate to utm
                    x, y = self.ll2xy(lon, lat)
                    p.append([x, y, z])
                    pll.append([lon, lat, z])
            elif inputCoordinates in ('xyz'):
                for j in range(1,nbrcorners+1):
                    x, y, z = L[i+j].split()
                    # Pass as floating point
                    x = float(x); y = float(y); z = float(z)
                    # translate to utm
                    lon, lat = self.xy2ll(x, y)
                    p.append([x, y, z])
                    pll.append([lon, lat, z])
                
            p = np.array(p)
            pll = np.array(pll)
            # Store these in the lists
            self.patch.append(p)
            self.patchll.append(pll)
            # increase i
            i += 5

        # Close the file
        fgfile.close()
        
        self.patch = np.array(self.patch)
        self.patchll = np.array(self.patchll)
        return
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    # Write patches to a GMT style file
    def writeFault2File(self,filename):
        '''
        Writes the patch corners in a file that can be used in psxyz.

        :Args:
            * filename : Name of the file
        '''
        if self.verbose:
            print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(os.path.join(self.savedir,filename), 'w')
        
        # Create the string for the color
        string = '  '
        
        # Loop over the patches
        nPatches = len(self.patch)
        for p in range(nPatches):
            # Write the string to file
            fout.write('> {} \n'.format(string))

            # Write the 4 patch corners (the order is to be GMT friendly)
            if self.inputCoordinates == 'lonlatz':
                p = self.patchll[p]
            elif self.inputCoordinates == 'xyz':
                p = self.patch[p]
            for i in range(len(p)):
                fout.write('{} {} {} \n'.format(p[i][0], p[i][1], p[i][2]))

        fout.close()
        return
        
    # ----------------------------------------------------------------------
    # Lon lat Transform
    def ll2xy(self, lon, lat):
        '''
        Do the lat/lon to UTM transform. 
        Input is in degrees. UTM coordinates are returned in km.

        :Args:
            * lon       : Longitude (deg)
            * lat       : Latitude (deg)

        :Returns:
            * x         : UTM coordinate x (km)
            * y         : UTM coordinate y (km)
        '''

        # Transpose 
        x, y = self.putm(lon, lat)

        # Put it in Km
        x = x/1000.
        y = y/1000.

        # All done
        return x, y
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # UTM transform
    def xy2ll(self, x, y):
        '''
        Do the UTm to lat/lon transform.
        Input is in km. Output is in degrees.

        :Args:
            * x         : UTM longitude (km).
            * y         : UTM latitude (km)

        :Returns: 
            * lon       : Longitude (degrees)
            * lat       : Latitude (degree)
        '''

        # Transpose and return
        return self.putm(x*1000., y*1000., inverse=True)
    # ---------------------------------------------------------------------- 
        
    # ----------------------------------------------------------------------         
    @staticmethod
    def toRadians(angle):
        return np.divide(np.dot(angle, np.pi), 180.0)
    # ---------------------------------------------------------------------- 
        
    # ----------------------------------------------------------------------
    @staticmethod
    def toDegrees(angle):
        return np.divide(np.dot(angle, 180.0), np.pi)
    # ---------------------------------------------------------------------- 
        
    # ----------------------------------------------------------------------  
    # Rotation of a point around an axis, in 3D
    def pointRotation3D(self, iPatch, iPoint, theta, p_axis1, p_axis2):
        '''
        Rotate a point with an arbitrary axis (fault tip)
        Used in rotatePatch
        
        :Args:
            * Point: array of coordinates for the patch corner (point) to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        :Returns:
            * rotated point
        Reference: 'Rotate A Point About An Arbitrary Axis (3D)' - Paul Bourke 
        '''
        
        point = self.patch[iPatch][iPoint]
        
        # Translate so axis is at origin    
        p = point - p_axis1
    
        N = p_axis2 - p_axis1
        Nm = np.sqrt(N[0]**2 + N[1]**2 + N[2]**2)
        
        # Rotation axis unit vector
        n = [N[0]/Nm, N[1]/Nm, N[2]/Nm]
    
        # Matrix common factors     
        c = np.cos(self.toRadians(theta))
        t = 1 - np.cos(self.toRadians(theta))
        s = np.sin(self.toRadians(theta))
        X = n[0]
        Y = n[1]
        Z = n[2]
    
        # Matrix 'M'
        d11 = t*X**2 + c
        d12 = t*X*Y - s*Z
        d13 = t*X*Z + s*Y
        d21 = t*X*Y + s*Z
        d22 = t*Y**2 + c
        d23 = t*Y*Z - s*X
        d31 = t*X*Z - s*Y
        d32 = t*Y*Z + s*X
        d33 = t*Z**2 + c
    
        #            |p.x|
        # Matrix 'M'*|p.y|
        #            |p.z|
        q = np.empty((3))
        q[0] = d11*p[0] + d12*p[1] + d13*p[2]
        q[1] = d21*p[0] + d22*p[1] + d23*p[2]
        q[2]= d31*p[0] + d32*p[1] + d33*p[2]
        
        # Translate axis and rotated point back to original location    
        return np.array(q + p_axis1)
    # ----------------------------------------------------------------------  
        
    # ----------------------------------------------------------------------
    # rotation of a fault patch
    def rotatePatch(self, iPatch , theta, p_axis1, p_axis2):
        '''
        Rotate a patch with an arbitrary axis (fault tip)
        Used by class uncertainties
        
        :Args:
            * iPatch: index of the patch to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        :Returns:
            * rotated patch
        '''
        if self.verbose:
            print('Rotating patch {} '.format(iPatch))
        
        # Calculate rotated patch
        rotated_patch = []
        for i in range(np.shape(self.patch[iPatch])[0]):
            rotated_patch.append(self.pointRotation3D(iPatch,i, theta, p_axis1, p_axis2))
                         
        # Replace
        self.patch[iPatch] = np.array(rotated_patch)
    
        # Build the ll patch
        patchll = []
        for i in range(np.shape(self.patch[iPatch])[0]):
            lon1, lat1 = self.xy2ll(self.patch[iPatch][i][0], self.patch[iPatch][i][1])
            z1 = self.patch[iPatch][i][2]
            patchll.append([lon1, lat1, z1])
    
        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
     # ----------------------------------------------------------------------
       
     # ----------------------------------------------------------------------
     # Translation of a fault patch
    def translatePatch(self, iPatch , tr_vector):
        '''
        Translate a patch
        Used by class uncertainties
        
        :Args:
            * iPatch: index of the patch to be rotated
            * tr_vector: array, translation vector in 3D
            
        :Returns:
            * None
        '''        
        # Calculate rotated patch
        tr_patch=[]
        for i in range(np.shape(self.patch[iPatch])[0]):
            tr_patch.append( [self.patch[iPatch][i][0]+tr_vector[0], 
                              self.patch[iPatch][i][1]+tr_vector[1], 
                              self.patch[iPatch][i][2]+tr_vector[2]])
                                                     
        # Replace
        self.patch[iPatch] = np.array(tr_patch)
    
        # Build the ll patch
        patchll = []
        for i in range(np.shape(self.patch[iPatch])[0]):
            lon1, lat1 = self.xy2ll(self.patch[iPatch][i][0], self.patch[iPatch][i][1])
            z1 = self.patch[iPatch][i][2]
            patchll.append([lon1, lat1, z1])

        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
    # ----------------------------------------------------------------------
 
    # ----------------------------------------------------------------------
    # Write new fault geometries with varying dip
    def varyDip(self, dip_range):
        '''
        Calculate new fault geometries with various dip values
        
        :Args:
            * dip_range : array, dip values (in degrees) to test around the prior fault dip
                    ex: dip_range = [-2,-1,0,1,2]
        :Returns:
            None
        '''
        # Calculate new faults
        if self.verbose: 
            print('---------------------------------')
            print('---------------------------------')
            print('Calculating new Fault Geometry')
            
        for d in dip_range: 
            #  Modify the fault geometry
            p_axis1 = np.array([self.patch[0][0][0],self.patch[0][0][1],self.patch[0][0][2]])
            p_axis2 = np.array([self.patch[self.nstrike][1][0],self.patch[self.nstrike][1][1],self.patch[self.nstrike][1][2]])
            
            # Rotate the patches
            for p in range(len(self.patch)):
                self.rotatePatch(p,d, p_axis1, p_axis2)  
            #write fault in file               
            self.writeFault2File('.'.join(self.filename.split('.')[:-1])+'_dip_'+str(d)+'.'+self.filename.split('.')[-1])    
        
        return
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    # Write new fault geometries with varying strike
    def varyStrike(self, stk_range, rotation_axis):
        '''
        Calculate new fault geometries with various strike values
        
        :Args:
            * strike_range : array, strike values (in degrees) to test around the prior fault strike
                    ex: strike_range = [-1,0,1,2]
             * rotation_axis : can be 'center', 'first', 'second' or an had-oc position [X,Y]
                    define the axis around which the fault will rotate
                    'center' : the center of the fault
                    'first'  : the first point of the fault trace (Fault.xf[0],Fault.yf[0])
                    'last'   : the last point of the fault trace (Fault.xf[-1],Fault.yf[-1])
                    [X,Y]    : ad-hoc coordinates
        :Returns:
            None
        '''
        # Define the rotation axis
        if rotation_axis == 'center':
            p_axis1 = np.array([(self.patch[0][0][0]+self.patch[self.nstrike][1][0])/2,(self.patch[0][0][1]+self.patch[self.nstrike][1][1])/2,0])
            p_axis2 = np.array([(self.patch[0][0][0]+self.patch[self.nstrike][1][0])/2,(self.patch[0][0][1]+self.patch[self.nstrike][1][1])/2,10])
        elif rotation_axis == 'first':
            p_axis1 = np.array([self.patch[0][0][0],self.patch[0][0][1],0])
            p_axis2 = np.array([self.patch[0][0][0],self.patch[0][0][1],10])
        elif rotation_axis == 'last':
            p_axis1 = np.array([self.patch[self.nstrike][1][0],self.patch[self.nstrike][1][1],0])
            p_axis2 = np.array([self.patch[self.nstrike][1][0],self.patch[self.nstrike][1][1],10])
        else:
            p_axis1 = np.array([rotation_axis[0],rotation_axis[1],0])
            p_axis2 = np.array([rotation_axis[0],rotation_axis[1],10])
            
        # Calculate new faults
        if self.verbose: 
            print('---------------------------------')
            print('---------------------------------')
            print('Calculating new Fault Geometry')
            
        for s in stk_range: 
            #  Modify the fault geometry
            for p in range(len(self.patch)):
                self.rotatePatch(p,s, p_axis1, p_axis2)  
            #write fault in file               
            self.writeFault2File('.'.join(self.filename.split('.')[:-1])+'_strike_'+str(s)+'.'+self.filename.split('.')[-1])    
        
        return
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    # Write new fault geometries with varying position
    def varyPosition(self, position_range):
        '''
        Calculate new fault geometries with various position values
        
        :Args:
            * position_range : array, position values (in km) to test around the prior fault position
                    Positive values are +90 degrees from the prior strike
                    ex: position_range = [-1,0,1,2]
        :Returns:
            None
        '''
        # Calculate new faults
        if self.verbose: 
            print('---------------------------------')
            print('---------------------------------')
            print('Calculating new Fault Geometry')
            
        for r in position_range: 
            # Get the fault strike (in radians)
            strike = np.arccos( (self.patch[self.nstrike][1][1]-self.patch[0][0][1]) / 
                     np.sqrt( (self.patch[self.nstrike][1][0]-self.patch[0][0][0])**2 
                     + (self.patch[self.nstrike][1][1]-self.patch[0][0][1])**2 ))  
                    
            # Calculate the translation vector
            x_translation = np.sin(np.pi/2 - strike)* r
            y_translation = np.cos(np.pi/2 - strike)* r
            z_translation = 0
            vector = [x_translation, y_translation, z_translation]
                
            #  Modify the fault geometry
            for p in range(len(self.patch)):
                self.translatePatch(p,vector)    
            #write fault in file               
            self.writeFault2File('.'.join(self.filename.split('.')[:-1])+'_position_'+str(r)+'.'+self.filename.split('.')[-1])    
        
        return
        

class uncertainties:  
    '''
    Class to calculate the covariance matrix of the predictions related to the uncertainties in the fault geometry.
    '''
    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, gfdir, Np, multifault=False, export=None, verbose=True):
        '''
        Class calculating the covariance matrix of the predictions.
    
        :Args:
            * gfdir      : string with path to directory where Green's Functions are stored   
            * Np         : number of parameters of the model
            * multifault : False if only 1 fault segment, True else
                    Faulse by default
            * export     : None by default, or string with path to directory
    
        '''
        
        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing uncertainties object")
            
        # Store things into self
        self.multifault = multifault
        self.export = export
        self.gfdir = gfdir
        self.Np = Np
               
        # Set the kernels and covariance matrices as empty arrays
        self.KDip = []
        self.KStrike = []
        self.KPosition = []
        self.KernelsFull = []
        
        self.CovDip = []
        self.CovStrike = []
        self.CovPosition = []
        self.CovFull = []
        
        self.CpDip = []
        self.CpStrike = []
        self.CpPosition = []
        self.CpFull = []
        self.CdFull = []
        
        return
    # ----------------------------------------------------------------------
       
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault dip 
    def calcCpDip(self, filenames, dip_range, sigma, mprior):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault dip.
        From Ragon et al. (2018) GJI
        
        :Args:
            * filenames : array with strings, corresponding to name of the Green's Functions files
                    ex: ['GF_dip_-1.txt','GF_dip_0.txt','GF_dip_1.txt']
                    If there are several fault segments (e.g. multifault is True), len(filenames) must be equal to the number of segments
                    ex: [['GF_seg1_dip_-1.txt','GF_seg1_dip_0.txt','GF_seg1_dip_1.txt'],['GF_seg2_dip_-2.txt','GF_seg2_dip_-1.txt','GF_seg2_dip_0.txt','GF_seg2_dip_1.txt']]
            * dip_range : difference between initial dip and the ones used to calculate Green's Functions (in degrees).
                    (the same dip_range as the one used to calculate Green's functions)
                    ex: dip_range = [-1,0,1,2]
                    If there are several fault segments (e.g. multifault is True), len(dip_range) must be equal to the number of segments
                    ex: dip_range = [[-1,0,1,2],[-2,-1,0,1,2,3],[-1.5,-1,-0.5,0],[0,1,2]] for a fault with 4 segments
            * sigma     : prior uncertainty (standard deviation) in the dip parameter (in degrees)
                    ex: sigma = 5
                    If there are several fault segments (e.g. multifault is True), len(sigma) must be equal to the number of segments
                    ex: sigma = [5,7,2,3] for a fault with 4 segments
            * mprior    : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
                
        :Returns:
            * CpDip
            
        '''
        
        # For a fault with one segment
        if self.multifault is False:
            # Read GFs 
            gfs = []
            for i in range(len(filenames)):
                gf = np.fromfile(os.path.join(self.gfdir,filenames[i]))
                gf = np.reshape(gf, (int(len(gf)/self.Np),self.Np))
                gfs.append(gf)
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            coeff = []
            for d in dip_range:
                if d < 0:
                    dip_vals = [d,0]
                elif d == 0:
                    continue
                else:
                    dip_vals=[0,d]
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dip_vals,[gfs[k][i,j] for k in [dip_range.index(dip_vals[0]),dip_range.index(dip_vals[1])] ])
                coeff.append(slope)
            
            # Select the maximum coefficient
            Kdip = np.max(coeff,axis=0)
            # Build the Covariance matrix                
            K = []
            K.append(Kdip)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            k = np.transpose(np.matmul(K, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpDip = np.matmul(C1, np.transpose(k))
            
            self.KDip = kernels
            self.CovDip = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KDip
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KDip))
            if self.CovFull==[]:
                self.CovFull = self.CovDip
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovDip)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovDip]]))
                
            self.CpDip = CpDip
            if self.CpFull==[]:
                self.CpFull = self.CpDip
            else:
                self.CpFull = np.add(self.CpFull, self.CpDip)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpDip.tofile(self.export+'CpDip.bin')
            
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpDip successfully calculated')
            
        # For a multi-segmented fault
        else:
            Kdip = []
            
            for f in range(len(filenames)):
                # Read GFs 
                gfs = []
                for i in range(len(filenames[f])):
                    gf = np.fromfile(os.path.join(self.gfdir,filenames[f][i]))
                    gf = np.reshape(gf, (len(gf)/self.Np,self.Np))
                    gfs.append(gf)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                coeff = []
                for d in dip_range:
                    if d < 0:
                        dip_vals = [d,0]
                    elif d == 0:
                        continue
                    else:
                        dip_vals=[0,d]
                    for i in range(gfs[0].shape[0]):
                        for j in range(gfs[0].shape[1]):
                            # Do a linear regression for each couple parameter/data
                            slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dip_vals,[gfs[k][i,j] for k in [dip_range.index(dip_vals[0]),dip_range.index(dip_vals[1])] ])
                    coeff.append(slope)
            
                # Select the maximum coefficient
                Kdip.append( np.max(coeff,axis=0) )
                
            # Build the Covariance matrix                
            kernels = np.asarray(Kdip)
            k = np.transpose(np.matmul(kernels, mprior))
            Covdip = np.zeros((len(filenames),len(filenames)))
            for f in range(len(filenames)):
                Covdip[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covdip)
            CpDip = np.matmul(C1, np.transpose(k))
            
            self.KDip = kernels
            self.CovDip = Covdip
            if self.KernelsFull==[]:
                self.KernelsFull = self.KDip
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KDip))
            if self.CovFull==[]:
                self.CovFull = self.CovDip
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovDip)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovDip]]))
                
            self.CpDip = CpDip
            if self.CpFull==[]:
                self.CpFull = self.CpDip
            else:
                self.CpFull = np.add(self.CpFull, self.CpDip)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpDip.tofile(self.export+'CpDip.bin')         
            
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpDip successfully calculated')
                
        return self.CpDip
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault strike    
    def calcCpStrike(self, filenames, strike_range, sigma, mprior):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault strike.
        From Ragon et al. (2018) GJI
        
        :Args:
            * filenames : array with strings, corresponding to name of the Green's Functions files
                    ex: ['GF_strike_-1.txt','GF_strike_0.txt','GF_strike_1.txt']
                    If there are several fault segments (e.g. multifault is True), len(filenames) must be equal to the number of segments
                    ex: [['GF_seg1_strike_-1.txt','GF_seg1_strike_0.txt','GF_seg1_strike_1.txt'],['GF_seg2_strike_-2.txt','GF_strike_dip_-1.txt','GF_seg2_strike_0.txt','GF_seg2_strike_1.txt']]
            * strike_range : array,difference between initial strike and the ones used to calculate Green's Functions (in degrees).
                    Positive angles are counter-clockwise looking toward the center of the Earth
                    ex: strike_range = [-1,0,1,2]
                    If whole_fault is False (e.g. multifault is True), len(strike_range) must be equal to len(filenames)
                    ex: strike_range= [[-1,0,1,2],[-2,-1,0,1,2,3],[-1.5,-1,-0.5,0],[0,1,2]] for a fault with 4 segments
            * sigma : prior uncertainty (standard deviation) in the strike parameter (in degrees)
                    ex: sigma = 5
                    If whole_fault is False (e.g. multifault is True), len(sigma) must be equal to len(filenames)
                    ex: sigma = [5,7,2,3] for a fault with 4 segments
            * mprior : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties

                
        :Returns:
            * CpStrike
        '''
        
        # For a fault with one segment
        if self.multifault is False:
            # Read GFs 
            gfs = []
            for i in range(len(filenames)):
                gf = np.fromfile(os.path.join(self.gfdir,filenames[i]))
                gf = np.reshape(gf, (len(gf)/self.Np,self.Np))
                gfs.append(gf)
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            coeff = []
            for s in strike_range:
                if s < 0:
                    strike_vals = [s,0]
                elif s == 0:
                    continue
                else:
                    strike_vals=[0,s]
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(strike_vals,[gfs[k][i,j] for k in [strike_range.index(strike_vals[0]),strike_range.index(strike_vals[1])] ])
                coeff.append(slope)
            
            # Select the maximum coefficient
            Kstk = np.max(coeff,axis=0)
            # Build the Covariance matrix
            K = []
            K.append(Kstk)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpStrike = np.matmul(C1, np.transpose(k))
            
            self.KStrike = kernels
            self.CovStrike = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KStrike
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KStrike))
            if self.CovFull==[]:
                self.CovFull = self.CovStrike
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovStrike)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovStrike]]))            
            
            self.CpStrike = CpStrike
            if self.CpFull==[]:
                self.CpFull = self.CpStrike
            else:
                self.CpFull = np.add(self.CpFull, self.CpStrike)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpStrike.tofile(self.export+'CpStrike.bin')
            
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpStrike successfully calculated')
            
        # For a multi-segmented fault
        else:
            Kstk = []
            
            for f in range(len(filenames)):
                # Read GFs 
                gfs = []
                for i in range(len(filenames[f])):
                    gf = np.fromfile(os.path.join(self.gfdir,filenames[f][i]))
                    gf = np.reshape(gf, (len(gf)/self.Np,self.Np))
                    gfs.append(gf)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                coeff = []
                for s in strike_range:
                    if s < 0:
                        strike_vals = [s,0]
                    elif s == 0:
                        continue
                    else:
                        strike_vals=[0,s]
                    for i in range(gfs[0].shape[0]):
                        for j in range(gfs[0].shape[1]):
                            # Do a linear regression for each couple parameter/data
                            slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(strike_vals,[gfs[k][i,j] for k in [strike_range.index(strike_vals[0]),strike_range.index(strike_vals[1])] ])
                    coeff.append(slope)
                
                # Select the maximum coefficient
                Kstk.append( np.max(coeff,axis=0) )
                
            # Build the Covariance matrix            
            kernels = np.asarray(Kstk)
            k = np.transpose(np.matmul(kernels, mprior))
            Covstk = np.zeros((len(filenames),len(filenames)))
            for f in range(len(filenames)):
                Covstk[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covstk)
            CpStrike = np.matmul(C1, np.transpose(k))
            
            self.KStrike = kernels
            self.CovStrike = Covstk
            if self.KernelsFull==[]:
                self.KernelsFull = self.KStrike
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KStrike))
            if self.CovFull==[]:
                self.CovFull = self.CovStrike
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovStrike)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovStrike]]))    
            
            self.CpStrike = CpStrike
            if self.CpFull==[]:
                self.CpFull = self.CpStrike
            else:
                self.CpFull = np.add(self.CpFull, self.CpStrike)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpStrike.tofile(self.export+'CpStrike.bin')         
            
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpStrike successfully calculated')
            
        return self.CpStrike
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault position
    def calcCpPosition(self, filenames, position_range, sigma, mprior):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault position.
        From Ragon et al. (2018) GJI
        
        :Args:
            * filenames : array with strings, corresponding to name of the Green's Functions files
                    ex: ['GF_strike_-1.txt','GF_strike_0.txt','GF_strike_1.txt']
                    If there are several fault segments (e.g. several faults in fault), len(filenames) must be equal to the number of segments
                    ex: [['GF_seg1_strike_-1.txt','GF_seg1_strike_0.txt','GF_seg1_strike_1.txt'],['GF_seg2_strike_-2.txt','GF_strike_dip_-1.txt','GF_seg2_strike_0.txt','GF_seg2_strike_1.txt']]
            * position_range : difference between initial position and the ones used to calculate Green's Functions (in km).
                    Positive values are +90 degrees from the prior strike
                    ex: position_range = [-1,0,1,2]
                    If there are several fault segments (e.g. several faults in fault), len(dip_range) must be equal to the number of segments
                    ex: position_range = [[-1,0,1,2],[-2,-1,0,1,2,3],[-1.5,-1,-0.5,0],[0,1,2]] for a fault with 4 segments
            * sigma : prior uncertainty (standard deviation) in the fault position (in km)
                    ex: sigma = 5
                    If there are several fault segments (e.g. multifault is True), len(sigma) must be equal to len(filenames)
                    ex: sigma = [1,0.5,1,0.5] for a fault with 4 segments
            * mprior : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
                   
        :Returns:
            * CpPosition
        '''
        # For a fault with one segment
        if self.multifault is False:
            # Read GFs 
            gfs = []
            for i in range(len(filenames)):
                gf = np.fromfile(os.path.join(self.gfdir,filenames[i]))
                gf = np.reshape(gf, (len(gf)/self.Np,self.Np))
                gfs.append(gf)
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            coeff = []
            for p in position_range:
                if p < 0:
                    position_vals = [p,0]
                elif p == 0:
                    continue
                else:
                    position_vals=[0,p]
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(position_vals,[gfs[k][i,j] for k in [position_range.index(position_vals[0]),position_range.index(position_vals[1])] ])
                coeff.append(slope)
            
            # Select the maximum coefficient
            Kpos = np.max(coeff,axis=0)
            # Build the Covariance matrix
            K = []
            K.append(Kpos)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpPosition = np.matmul(C1, np.transpose(k))
            
            self.KPosition = kernels
            self.CovPosition = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KPosition
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KPosition))
            if self.CovFull==[]:
                self.CovFull = self.CovPosition
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovPosition)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovPosition]]))                
            
            self.CpPosition = CpPosition
            if self.CpFull==[]:
                self.CpFull = self.CpPosition
            else:
                self.CpFull = np.add(self.CpFull, self.CpPosition)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpPosition.tofile(self.export+'CpPosition.bin') 
                
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpPosition successfully calculated')
            
        # For a multi-segmented fault
        else:
            Kpos = []
            
            for f in range(len(filenames)):
                # Read GFs 
                gfs = []
                for i in range(len(filenames[f])):
                    gf = np.fromfile(os.path.join(self.gfdir,filenames[f][i]))
                    gf = np.reshape(gf, (len(gf)/self.Np,self.Np))
                    gfs.append(gf)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                coeff = []
                for p in position_range:
                    if p < 0:
                        position_vals = [p,0]
                    elif p == 0:
                        continue
                    else:
                        position_vals=[0,p]
                    for i in range(gfs[0].shape[0]):
                        for j in range(gfs[0].shape[1]):
                            # Do a linear regression for each couple parameter/data
                            slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(position_vals,[gfs[k][i,j] for k in [position_range.index(position_vals[0]),position_range.index(position_vals[1])] ])
                    coeff.append(slope)
                
                # Select the maximum coefficient
                Kpos.append( np.max(coeff,axis=0) )
            
            # Build the Covariance matrix
            kernels = np.asarray(Kpos)
            k = np.transpose(np.matmul(kernels, mprior))
            Covpos = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covpos[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covpos)
            CpPosition = np.matmul(C1, np.transpose(k))
            
            self.KPosition = kernels
            self.CovPosition = Covpos
            if self.KernelsFull==[]:
                self.KernelsFull = self.KPosition
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KPosition))
            if self.CovFull==[]:
                self.CovFull = self.CovPosition
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovPosition)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovPosition]]))                  
            
            self.CpPosition = CpPosition
            if self.CpFull==[]:
                self.CpFull = self.CpPosition
            else:
                self.CpFull = np.add(self.CpFull, self.CpPosition)
            
            if self.export is not None:
                self.CpFull.tofile(self.export+'CpFull.bin')
                self.CpPosition.tofile(self.export+'CpPosition.bin')         
            
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print('CpPosition successfully calculated')
                
        return self.CpPosition
    # ----------------------------------------------------------------------
    

#EOF          