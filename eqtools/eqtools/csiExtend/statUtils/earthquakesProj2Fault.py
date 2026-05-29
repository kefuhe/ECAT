from csi import seismiclocations
from csi import RectangularPatches as Rect
from csi import TriangularPatches as Tri
from csi import SourceInv

import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import os
import copy


def _split_patch_batch_helper(args):
    """Helper function for parallel patch splitting"""
    patch_batch, receiver = args
    results = []
    for patch in patch_batch:
        p1, p2, p3, p4 = receiver.splitPatch(patch)
        results.extend([p1, p2, p3, p4])
    return results


class EqseqProj2Fault(SourceInv):
    '''
    '''
    
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
        self.receiver = receiver
    
    def setSeismic(self, seismic):
        self.seismic = seismic
    
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

        # All Done
        return
    
    def setOutputfile(self, outputfile):
        self.outputfile = outputfile

    # def splitReceiver(self, num_splits):
    #     """
    #     Split the receiver patches into smaller subpatches and store the original indices.
    
    #     Args:
    #         num_splits (int): The number of times to split each patch.
    
    #     Returns:
    #         None
    #     """
    #     receiver = self.receiver.duplicateFault()
    #     original_indices = list(range(len(self.receiver.patch)))  # Initialize with original indices
    
    #     for _ in range(num_splits):
    #         subpatches = []
    #         new_indices = []
    #         for ip, original_index in enumerate(original_indices):
    #             ipatch = receiver.patch[ip]
    #             p1, p2, p3, p4 = receiver.splitPatch(ipatch)
    #             subpatches.extend(np.array([p1, p2, p3, p4]))
    #             new_indices.extend([original_index, original_index, original_index, original_index])  # Store the original index for each new subpatch
    #         receiver.patch = subpatches
    #         original_indices = new_indices  # Update the original indices list
    #         # receiver.computeEquivRectangle()
    #         receiver.patch2ll()
    #         receiver.initializeslip()
    #         # Too slow
    #         # receiver.setVerticesFromPatches()
    
    #     self.receiver_dense = receiver
    #     self.receiver_dense.original_indices = original_indices  # Store the original indices in the receiver_dense
    
    #     # All Done
    #     return

    def splitReceiver(self, num_splits, use_parallel=False, n_workers=None):
        """
        Split the receiver patches with optional parallel processing.
        """
        if use_parallel and num_splits > 1:
            return self._splitReceiver_parallel(num_splits, n_workers)
        else:
            return self._splitReceiver_sequential(num_splits)
    
    def _splitReceiver_parallel(self, num_splits, n_workers=None):
        """
        Parallel version of splitReceiver using multiprocessing with progress bar.
        """
        from multiprocessing import Pool, cpu_count
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("tqdm not available, using simple progress indication")
        
        if n_workers is None:
            n_workers = min(cpu_count(), 8)
        
        receiver = self.receiver.duplicateFault()
        original_indices = np.arange(len(self.receiver.patch))
        
        print(f"Starting parallel patch splitting with {n_workers} workers...")
        print(f"Initial patches: {len(self.receiver.patch)}")
        
        # Progress bar for splits
        split_iterator = tqdm(range(num_splits), desc="Splitting") if use_tqdm else range(num_splits)
        
        for split_idx in split_iterator:
            patches = receiver.patch
            num_patches = len(patches)
            
            if not use_tqdm:
                print(f"\n--- Split {split_idx + 1}/{num_splits} ---")
                print(f"Processing {num_patches} patches...")
    
            # Batch processing
            batch_size = max(1, num_patches // n_workers)
            patch_batches = [patches[i:i+batch_size] 
                            for i in range(0, num_patches, batch_size)]
    
            # Prepare arguments for parallel processing
            args_list = [(batch, receiver) for batch in patch_batches]
    
            # Parallel processing
            with Pool(n_workers) as pool:
                if use_tqdm:
                    batch_results = list(tqdm(
                        pool.imap(_split_patch_batch_helper, args_list), 
                        total=len(args_list), 
                        desc=f"Split {split_idx+1} batches",
                        leave=False
                    ))
                else:
                    batch_results = pool.map(_split_patch_batch_helper, args_list)
                    print("[v] Parallel processing completed")
    
            # Merge results
            new_patches = []
            batch_iterator = tqdm(batch_results, desc="Merging", leave=False) if use_tqdm else batch_results
            
            for batch_result in batch_iterator:
                new_patches.extend(batch_result)
            
            receiver.patch = new_patches
            original_indices = np.repeat(original_indices, 4)
            
            if use_tqdm:
                split_iterator.set_postfix(patches=len(receiver.patch))
            else:
                print(f"Split {split_idx + 1} completed: {len(receiver.patch)} patches")
        
        print(f"\nFinalizing... Converting {len(receiver.patch)} patches to numpy array")
        receiver.patch = np.array(receiver.patch)
        
        print("Converting patches to lat/lon coordinates...")
        receiver.patch2ll()
        
        print("Initializing slip values...")
        receiver.initializeslip()
        
        self.receiver_dense = receiver
        self.receiver_dense.original_indices = original_indices.tolist()
        
        print(f"[v] Parallel splitting completed successfully!")
        print(f"Final result: {len(receiver.patch)} patches from {len(self.receiver.patch)} original patches")
        
        return
    
    def _splitReceiver_sequential(self, num_splits):
        """
        Split the receiver patches into smaller subpatches and store the original indices.
        Optimized version with pre-allocation.
    
        Args:
            num_splits (int): The number of times to split each patch.
    
        Returns:
            None
        """
        receiver = self.receiver.duplicateFault()
        
        # Pre-calculate the number of patches after all splits
        initial_patch_count = len(self.receiver.patch)
        final_patch_count = initial_patch_count * (4 ** num_splits)

        # Pre-allocate arrays
        final_indices = np.zeros(final_patch_count, dtype=int)

        # Initialize indices
        current_indices = np.arange(initial_patch_count)
        
        for split_idx in range(num_splits):
            num_current_patches = len(receiver.patch)
            new_patches = []

            # Batch processing
            for ip in range(num_current_patches):
                ipatch = receiver.patch[ip]
                p1, p2, p3, p4 = receiver.splitPatch(ipatch)
                new_patches.extend([p1, p2, p3, p4])

            # Update indices: each original index is copied 4 times
            current_indices = np.repeat(current_indices, 4)
            receiver.patch = new_patches

            # Only update other attributes after the last iteration
            if split_idx == num_splits - 1:
                # Convert patch list to numpy array before calling patch2ll()
                receiver.patch = np.array(receiver.patch)
                receiver.patch2ll()
                receiver.initializeslip()
    
        self.receiver_dense = receiver
        self.receiver_dense.original_indices = current_indices.tolist()
        
        return
    
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
        """
        Calculate the projection of seismic events onto the fault patches and store the original indices.
    
        Args:
            write2file (bool): Whether to write the projection results to a file (default is False).
            remove_fault_edge_events (bool): Whether to remove seismic events on the fault edge triangles (default is False).
    
        Returns:
            None
        """
        seis = self.seismic
        receiver = self.receiver_dense
    
        # The patch number nearest to the fault is obtained successively
        ipatch = seis.getClosestFaultPatch(receiver)
    
        # Create a list of patch centers
        Centers = [receiver.getpatchgeometry(i, center=True)[:3] for i in ipatch]
    
        seis_llh = []
        for i in range(len(Centers)):
            lon, lat = receiver.xy2ll(Centers[i][0], Centers[i][1])
            seis_llh.append([lon, lat, Centers[i][2]])
    
        seis_llh = pd.DataFrame(seis_llh, columns='lon lat dep'.split())
    
        seis_proj = copy.deepcopy(self.seismic)
        seis_proj.lon = seis_llh.lon.values
        seis_proj.lat = seis_llh.lat.values
        seis_proj.depth = seis_llh.dep.values
        seis_proj.lonlat2xy()
    
        # Store the original indices of the patches
        seis_proj.original_patch_indices = [receiver.original_indices[i] for i in ipatch]
    
        # Optionally remove seismic events on the fault edge triangles
        if remove_fault_edge_events:
            # Too slow
            receiver.setVerticesFromPatches()
            receiver.find_fault_edge_vertices(refind=True)
            edge_patches = []
            for iedge in ['left', 'right', 'bottom']:
                iedge_triangles = receiver.edge_triangles_indices[iedge]
                if isinstance(iedge_triangles, list):
                    edge_patches.extend(iedge_triangles)
                else:
                    edge_patches.extend(iedge_triangles.tolist())
            edge_patches = np.unique(edge_patches)

            mask = ~np.isin(ipatch, edge_patches)
            seis_proj.lon = seis_proj.lon[mask]
            seis_proj.lat = seis_proj.lat[mask]
            seis_proj.depth = seis_proj.depth[mask]
            seis_proj.original_patch_indices = np.array(seis_proj.original_patch_indices)[mask].tolist()
    
        self.seis_proj = seis_proj
    
        if write2file:
            self.write2file(self.outputfile)
    
        # All Done 
        return
    
    def calculate_seismic_projection_statistics(self):
        """
        Calculate the statistics of seismic event projections onto the fault patches.
    
        Returns:
            None
        """
        receiver = self.receiver
        seis_proj = self.seis_proj
    
        # Initialize slip to zero
        receiver.slip = np.zeros((len(receiver.patch), 3))
    
        # Count the number of seismic events in each patch using numpy's advanced indexing
        unique_indices, counts = np.unique(seis_proj.original_patch_indices, return_counts=True)
        receiver.slip[unique_indices, 0] = counts
        receiver.slip[:, 1:] = 0.0  # Set the dip and open slip to zero
    
        # All Done
        return
    
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
        from ...plottools import plot_slip_distribution
        self.calculate_seismic_projection_statistics()
    
        # Plot the statistics
        figname = f'seismic_stat_in_{self.receiver.name}' if figname is None else figname
        plot_slip_distribution(self.receiver, add_faults=[self.receiver], slip='strikeslip', cmap=cmap, norm=norm, figsize=figsize,
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
        self.calculate_seismic_projection_statistics()
    
        receiver = self.receiver
    
        # Save the statistics to a file
        if filename is None:
            filename = f'seismic_stat_in_{self.receiver.name}.gmt'
        receiver.writePatches2File(filename, add_slip='total')
    
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