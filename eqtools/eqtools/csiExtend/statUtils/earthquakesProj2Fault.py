from csi import seismiclocations
from csi import TriangularPatches as Tri
from csi import SourceInv

import pandas as pd
import numpy as np
import os
import copy


class EqseqProj2Fault(SourceInv):
    """Project a seismic catalog onto a CSI triangular receiver and count events.

    CSI owns complete mesh subdivision and boundary state. This object owns
    event assignments and analysis results; it never writes counts into slip.
    """
    
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 seismic=None, receiver=None, outputfile='earthquakes_projed.dat'):
        super().__init__(name, utmzone, ellps, lon0, lat0)
        if seismic is None:
            self.seismic = seismiclocations('seismic', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.seismic = seismic
        
        if receiver is None:
            self.receiver = Tri('receiver', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.receiver = receiver
        
        self.outputfile = outputfile
    
    def setReceiver(self, receiver):
        """Replace the input fault and discard results tied to its face IDs."""
        self.receiver = receiver
        self.__dict__.pop('receiver_dense', None)
        self.__dict__.pop('seis_proj', None)
    
    def setSeismic(self, seismic):
        """Replace the catalog; the existing dense receiver remains reusable."""
        self.seismic = seismic
        self.__dict__.pop('seis_proj', None)
    
    def setSeismicFromFile(self, seisfile, header=0):
        '''
          Y  m  D  H  M  Sec          Lat   Lon      Dep    Mag
        '''
        seis_info = pd.read_csv(seisfile, sep=r'\s+', skiprows=header, comment='#')
        datetime = seis_info.apply(lambda x: '{0:4d}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:.3f}'.format(*[int(i) for i in x.values[0:5]], x.values[6]), axis=1)
        seis_info['datetime'] = pd.to_datetime(datetime, format='%Y-%m-%dT%H:%M:%S.%f')

        seis = self.seismic

        seis.time = seis_info.datetime.values
        seis.lon = seis_info.Lon.values
        seis.lat = seis_info.Lat.values
        seis.mag = seis_info.Mag.values
        seis.depth = seis_info.Dep.values
        seis.lonlat2xy()
        self.__dict__.pop("seis_proj", None)

        # All Done
        return
    
    def setOutputfile(self, outputfile):
        self.outputfile = outputfile

    def splitReceiver(self, num_splits):
        """Prepare an independent, complete receiver mesh for projection.

        ``num_splits`` is a non-negative integer. CSI supplies shared midpoint
        IDs and input-face lineage. Only the existing ``original_indices``
        attribute stores that lineage here; it is not duplicated on this object.
        Positive levels retain the historical zero-slip receiver convention.
        The vectorized kernel replaces the former multiprocessing interface.
        """
        subdivide = getattr(self.receiver, 'subdivided_copy', None)
        if not callable(subdivide):
            raise TypeError(
                "A CSI triangular receiver with subdivided_copy() is required; "
                "update the CSI fork and eqtools together."
            )
        receiver, parent_face_ids = subdivide(num_splits)
        receiver.original_indices = parent_face_ids.tolist()
        # Commit only after successful construction; old results survive failure.
        self.receiver_dense = receiver
        self.__dict__.pop('seis_proj', None)

    def write2file(self, outputfile=None, cotime=None):
        if outputfile is not None:
            self.outputfile = outputfile
        
        eqtime = self.seis_proj.time
        if cotime is not None:
            dt = eqtime - cotime
            dt_days = dt.apply(lambda x: x.days + x.seconds/3600./24.)
            # 投影信息输出
            self.seis_proj.write2file(outputfile, add_column=dt_days)
        else:
            self.seis_proj.write2file(outputfile)

        # All Done
        return 

    def calproj(self, write2file=False, remove_fault_edge_events=False):
        """Project events to nearest fine-face centers without changing inputs.

        ``splitReceiver`` must run first, including ``splitReceiver(0)`` when no
        refinement is wanted. ``ipatch`` remains a local event-to-fine-face map.
        The projected catalog stores event-to-input-face IDs as
        ``original_patch_indices``. Boundary filtering removes assignments to
        fine faces on left/right/bottom; the top edge is retained.
        """
        if not hasattr(self, 'receiver_dense'):
            raise ValueError("Call splitReceiver(num_splits) before calproj().")
        seis = self.seismic
        receiver = self.receiver_dense

        # Keep CSI's nearest-center assignment and its historical tie behavior.
        if len(seis.lon):
            ipatch = np.asarray(seis.getClosestFaultPatch(receiver), dtype=int)
        else:
            ipatch = np.empty(0, dtype=int)
        centers = np.asarray([
            receiver.getpatchgeometry(i, center=True)[:3] for i in ipatch
        ]).reshape(-1, 3)
        lon, lat = receiver.xy2ll(centers[:, 0], centers[:, 1])
        original_patch_indices = np.asarray(receiver.original_indices, dtype=int)[ipatch]

        seis_proj = copy.deepcopy(seis)
        seis_proj.lon = np.asarray(lon)
        seis_proj.lat = np.asarray(lat)
        seis_proj.depth = centers[:, 2].copy()
        seis_proj.lonlat2xy()

        if remove_fault_edge_events:
            # The mesh is already complete. CSI reuses its valid boundary cache;
            # rebuilding connectivity from patch coordinates would break it.
            receiver.find_fault_fouredge_vertices(refind=False)
            edge_patches = np.unique(np.concatenate([
                np.asarray(receiver.edge_triangles_indices[edge], dtype=int)
                for edge in ('left', 'right', 'bottom')
            ]))
            mask = ~np.isin(ipatch, edge_patches)
            # CSI owns synchronized selection of all standard catalog fields.
            # This extension owns the custom event-to-input-face metadata.
            seis_proj._select(mask)
            original_patch_indices = original_patch_indices[mask]

        seis_proj.original_patch_indices = original_patch_indices.tolist()
        self.seis_proj = seis_proj
        if write2file:
            self.write2file(self.outputfile)

    def calculate_seismic_projection_statistics(self):
        """Return integer event counts in input-face order, including zeros.

        Count only events retained in ``seis_proj``. The returned array has one
        entry per original receiver patch; physical ``receiver.slip`` is never
        used as statistical storage or changed by this operation.
        """
        if not hasattr(self, 'seis_proj'):
            raise ValueError("Call calproj() before calculating event counts.")
        original_patch_indices = np.asarray(
            self.seis_proj.original_patch_indices, dtype=int,
        )
        return np.bincount(original_patch_indices, minlength=len(self.receiver.patch))

    def plot_seismic_projection_statistics(self, norm=None, cmap='precip3_16lev_change.cpt', figsize=(None, None),
                                           drawCoastlines=False, cbaxis=[0.1, 0.2, 0.1, 0.02],
                                           cblabel='', show=True, savefig=False, ftype='pdf', dpi=600, bbox_inches=None,
                                           remove_direction_labels=False, cbticks=None, cblinewidth=None, cbfontsize=None,
                                           cb_label_side='opposite', map_cbaxis=None, style=['notebook'], xlabelpad=None,
                                           ylabelpad=None, zlabelpad=None, xtickpad=None, ytickpad=None, ztickpad=None,
                                           elevation=None, azimuth=None, shape=(1.0, 1.0, 0.4), zratio=None, plotTrace=True,
                                           depth=None, zticks=None, map_expand=0.2, fault_expand=0.1, plot_faultEdges=False,
                                           faultEdges_color='k', faultEdges_linewidth=1.0, suffix='', show_grid=True,
                                           grid_color='#bebebe', background_color='white', axis_color=None,
                                           zaxis_position='bottom-left', figname=None):
        """
        Plot the statistics of seismic event projections onto the fault patches.
    
        Args:
            norm (tuple): The normalization range for the color scale.
            fignames (str): The name of the figure to save.
    
        Returns:
            None
        """
        from ...viztools import plot_slip_distribution
        counts = self.calculate_seismic_projection_statistics()
    
        # Plot the statistics
        figname = f'seismic_stat_in_{self.receiver.name}' if figname is None else figname
        plot_slip_distribution(self.receiver, add_faults=[self.receiver], slip=counts, cmap=cmap, norm=norm, figsize=figsize,
                               drawCoastlines=drawCoastlines, plot_on_2d=False, cbaxis=cbaxis, cblabel=cblabel,
                               show=show, savefig=savefig, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches,
                               remove_direction_labels=remove_direction_labels, cbticks=cbticks, cblinewidth=cblinewidth,
                               cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis, style=style,
                               xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad, xtickpad=xtickpad,
                               ytickpad=ytickpad, ztickpad=ztickpad, elevation=elevation, azimuth=azimuth, shape=shape,
                               zratio=zratio, plotTrace=plotTrace, depth=depth, zticks=zticks, map_expand=map_expand,
                               fault_expand=fault_expand, plot_faultEdges=plot_faultEdges, faultEdges_color=faultEdges_color,
                               faultEdges_linewidth=faultEdges_linewidth, suffix=suffix, show_grid=show_grid,
                               grid_color=grid_color, background_color=background_color, axis_color=axis_color,
                               zaxis_position=zaxis_position, figname=figname)
    
        # All Done
        return
    
    def save_seismic_projection_statistics(self, filename=None):
        """
        Save the statistics of seismic event projections onto the fault patches to a file.
    
        Args:
            filename (str): The name of the file to save the statistics.
    
        Returns:
            None
        """
        counts = self.calculate_seismic_projection_statistics()
    
        receiver = self.receiver
    
        # Save the statistics to a file
        if filename is None:
            filename = f'seismic_stat_in_{self.receiver.name}.gmt'
        receiver.writePatches2File(filename, add_slip=counts)
    
        # All Done
        return


if __name__ == '__main__':

    # 投影信息
    lon0, lat0 = 101.5, 37.5

    # Building the receiver fault
    slipfile = r'slip_total_0.gmt'
    outfile = os.path.join('.', 'seis_reloc_proj_test.gmt')
    receiver = Tri('Menyuan', utmzone=None, ellps='WGS84', verbose=True, lon0=lon0, lat0=lat0)
    receiver.readPatchesFromFile(slipfile, readpatchindex=False)

    # Build a seismiclocations object
    ## Case 1
    seisfile = r'relocated_seismic.txt'
    # seis = seismiclocations('seis_reloc', lon0=lon0, lat0=lat0)
    # seis.read_from_Hauksson(seisfile, header=4)
    # lon, lat = seis.lat, seis.lon
    # seis.lon = lon
    # seis.lat = lat
    # seis.lonlat2xy()

    # Define the projection object
    eqprojobj = EqseqProj2Fault('eqproj', utmzone=None, ellps='WGS84', 
                                lon0=lon0, lat0=lat0, receiver=receiver, outputfile=outfile)
    # Dense receiver
    eqprojobj.splitReceiver(3)

    # eqprojobj.setSeismic(seis)

    # Case 2
    eqprojobj.setSeismicFromFile(seisfile, header=3)

    # Select the time range and magnitude range, as well as the space range
    # minlon, maxlon, minlat, maxlat = [101.008153, 101.203804, 37.629900, 37.822300]
    # seis.selectbox(minlon, maxlon, minlat, maxlat, depth=100000., mindep=0.0)
    # seis.selecttime(start=[2001, 1, 1], end=[2101, 1, 1])
    eqprojobj.seismic.selectmagnitude(minimum=0, maximum=6.8)

    # Projection and output
    eqprojobj.calproj(write2file=True)
