"""Bathymetry loading, interpolation and smoothing (original algorithms preserved)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator
from numba import jit, njit

class DepthHandler:
    '''
    Loads data, interpolates it to the grid and smooths it to some rx0 value
    '''
    @property
    def xy_grid_at_depth_points(self):
        '''Projects mesh to depth-file coordinates'''
        if self.target_projection != self.depth_projection:
            return np.array([*self.Proj_depth(self.lon, self.lat)]).T
        else:
            return np.array([self.x, self.y]).T

    @property
    def x_d(self):
        '''mesh x-position in depth-file coordinates'''
        return self.xy_grid_at_depth_points[:,0]
    
    @property
    def y_d(self):
        '''mesh y-position in depth-file coordinates'''
        return self.xy_grid_at_depth_points[:,1]

    @property
    def h_raw(self):
        '''bathymetry directly interpolated from the source'''
        return self._h_raw
    
    @h_raw.setter
    def h_raw(self, var):
        self._h_raw = var    

    def make_depth(self):
        """Interpolate bathymetry and report where source coverage is missing.

        Numerical behavior for finite input is unchanged.  Invalid interpolated
        depths are still replaced by ``min_depth``, but their original locations
        are retained for interactive quality control.
        """
        constant_depth = type(self.depth_file) in (int, float)
        if constant_depth:
            # Preserve the legacy constant-depth path, including its treatment
            # of values below min_depth.
            h_raw = self.depth_file * np.ones(self.x.shape)
        else:
            if str(self.depth_file).lower().endswith('.npy'):
                from fvtools.prepro._buildcase.tiled_bathymetry import interpolate_unstructured_npy
                print('  - Memory-limited unstructured bathymetry interpolation')
                h_raw = interpolate_unstructured_npy(
                    self.depth_file, self.xy_grid_at_depth_points
                )
            else:
                depth = self.load_textbath(self.depth_file)
                depth = self.crop(depth)
                h_raw = self.interpolate_to_mesh(depth)

        h_raw = np.asarray(h_raw)
        if h_raw.ndim != 1 or len(h_raw) != len(self.x):
            raise ValueError('Interpolated bathymetry must have one value per grid node')

        # Take the mask BEFORE replacing missing values.  A node whose real
        # bathymetry happens to equal min_depth must not be labelled missing.
        missing_mask = ~np.isfinite(h_raw)
        missing_count = int(np.count_nonzero(missing_mask))
        if not constant_depth:
            # Same in-place minimum-depth rule as the previous BuildCase.
            np.maximum(h_raw, self.min_depth, out=h_raw)
        if missing_count:
            h_raw[missing_mask] = self.min_depth

        total = len(h_raw)
        at_min_depth_count = int(np.count_nonzero(h_raw == self.min_depth))
        self.bathymetry_missing_mask = missing_mask
        self.bathymetry_qc = {
            'total_nodes': total,
            'missing_nodes': missing_count,
            'missing_percent': 100.0 * missing_count / total if total else 0.0,
            'at_min_depth_nodes': at_min_depth_count,
            'min_depth': float(self.min_depth),
        }
        print(f'  - Bathymetry coverage: {missing_count:,}/{total:,} nodes '
              f'({self.bathymetry_qc["missing_percent"]:.2f}%) missing source depth')
        if missing_count:
            print(f'  - Replaced missing bathymetry at {missing_count:,} nodes '
                  f'with min_depth={self.min_depth:g} m; locations are shown on the QC map')
        self.h_raw = h_raw

    @staticmethod
    def load_numpybath(dptfile):
        """Load the bathymetry from a big numpy file"""
        depth_data = np.load(dptfile, allow_pickle = True)
        return depth_data

    @staticmethod
    def load_textbath(dptfile):
        """
        Load the depth file
        - .txt files are "raw", ie. has not been processed by any BuildCase runs before. (slow to load)
        - .npy files are already in a BuildCase friendly format, and don't take too long to load
        """
        # Load raw data
        print(f'  -> Load: {dptfile}')
        try:
            depth_data = np.loadtxt(dptfile, delimiter=',')
        except:
          try:
              depth_data = np.loadtxt(dptfile)
          except:
            try:
                depth_data = np.loadtxt(dptfile, delimiter=' ')
            except:
                depth_data = np.loadtxt(dptfile, skiprows = 1, delimiter = ',', usecols = [0,1,2])
        numpy_bath_name = f'{dptfile.replace(".txt", "").replace(".csv", "")}.npy'
        print(f'- Storing the full bathymetry as: {numpy_bath_name}')
        np.save(numpy_bath_name, depth_data)
        return depth_data

    def crop(self, depth_data):
        """
        Crop the depth file to your domain
        """
        if max(depth_data[:,0]) < max(self.x_d) or min(depth_data[:,0]) > min(self.x_d) or max(depth_data[:,1]) < max(self.y_d) or min(depth_data[:,1]) > min(self.y_d):
            plt.plot(depth_data[:,0], depth_data[:,1], 'r.', label = 'depth data')
            plt.triplot(self.x_d, self.y_d, self.tri, label = 'FVCOM ', zorder = 10)
            plt.legend()
            plt.show(block=False)
            raise ValueError('The bathymetry file does not cover the model domain!')

        print('  - Crop the bathymetry data')
        ind1 = np.logical_and(depth_data[:, 0] >= min(self.x_d) - 5000.0, depth_data[:, 0] <= max(self.x_d) + 5000.0)
        ind2 = np.logical_and(depth_data[:, 1] >= min(self.y_d) - 5000.0, depth_data[:, 1] <= max(self.y_d) + 5000.0)
        ind  = np.logical_and(ind1, ind2)

        depth = {}
        depth['x'] = depth_data[ind,0]
        depth['y'] = depth_data[ind,1]
        depth['h'] = depth_data[ind,2]
        return depth

    def interpolate_to_mesh(self, depth):
        """
        Interpolate data from the unstructured data array (data) to the unstructured mesh (M)
        """
        print('  - Build interpolator')
        point = np.array([depth['x'], depth['y']]).T
        interpolant = LinearNDInterpolator(point, depth['h'])

        print('  - Interpolate topography to nodes')
        h_raw = interpolant(self.x_d, self.y_d)

        i = np.where(h_raw[:] < self.min_depth)[0]
        if i.size:
            print(f'  - The depth at {len(i)} points was set equal to {self.min_depth} m')
        h_raw[i] = self.min_depth
        return h_raw

    def smooth_topo(self):
        """
        Smooth the topography where the smoothness is bad
        """
        # Initial pass to do the initial smoothing (reduce dataset noise)
        print('  - Reduce noise')
        print('    - Laplacian filter')
        self.h = self.laplacian_filter(self.x, self.y, self.h, self.nbsn, self.ntsn, self.SmoothFactor)

        print('  - Adjust slope')
        print('    - Mellor, Ezer and Oey scheme')
        i = 0
        while True:
            i += 1
            self.h, rx0_max, corrected = self.mellor_ezer_oey(self.h, self.ntsn, self.nbsn, rx0max = self.rx0max - 0.02)
            print(f'    {i}: Max rx0: {rx0_max:.2f} - Number of adjustments: {corrected}')
            if abs(rx0_max - self.rx0max) < self.rx0max*0.01 or rx0_max == 0:
                print('  - Bathymetry smoothed.')
                break

    @staticmethod
    @njit
    def laplacian_filter(x, y, h, nbsn, ntsn, SmoothFactor):
        """
        A simple lapacian filter to smooth the topography
        """
        h_smooth = np.copy(h)
        for node in range(len(x)):
            nodes  = nbsn[node,:ntsn[node]]
            smooth = np.mean(h[nodes])
            h_smooth[node] = (1-SmoothFactor)*h_smooth[node] + SmoothFactor*smooth
        return h_smooth

    @staticmethod
    @jit(nopython = True)
    def mellor_ezer_oey(raw_bath, ntsn, nbsn, rx0max = 0.2, nodes = None):
        """
        Parameters:
        -----------
        bathymetry, number of surrounding nodes, index of surrounding nodes, smoothing goal and
        what nodes to inspect

        Returns:
        -----------
        Smoothed bathymetry

        ===================================================================================
        Mellor, Ezer and Oey:
        The Pressure Gradient Conundrum of Sigma Coordinate Ocean Models
        J. Atmos. Oceanic Technol. (1994)
        https://doi.org/10.1175/1520-0426(1994)011%3C1126:TPGCOS%3E2.0.CO;2
        ===================================================================================
        """

        # initialize
        if nodes is None:
            nodes = np.arange(len(raw_bath))

        # Iterate over the mesh
        rmax, bath, corrected = 0, np.copy(raw_bath), 0
        for i in nodes:
            diff = bath[i] - bath[nbsn[i,:ntsn[i]]]
            nbsn_ind  = np.where(diff == np.max(diff))[0][0]
            ind  = nbsn[i,nbsn_ind]
            difference = diff[nbsn_ind]
            if difference < 0:
                continue

            ph = bath[i] + bath[ind]

            r  = diff[nbsn_ind]/ph
            if r > rx0max:
                corrected += 1
                rarray     = np.array((r,rmax))
                rmax       = np.max(rarray)
                delta      = 0.5*(bath[i]-bath[ind]-rx0max*(bath[i]+bath[ind]))
                bath[i]   -= delta
                bath[ind] += delta

        return bath, rmax, corrected

    def setup_metrics(self):
        """
        Setup metrics for secondary connectivity (nodes surrounding nodes). These are needed when smoothing the topography.
        """
        # Use TGE's topological connectivity without triggering its full
        # control-volume/gradient calculation through self.T.
        from fvtools.grid import tge

        n_nodes = len(self.x)
        nv = tge.check_nv(np.asarray(self.tri), self.x, self.y)
        n_cells = len(nv)
        if not n_cells or not n_nodes:
            raise ValueError('Cannot build smoothing neighbors for an empty mesh')

        nbe = tge.get_NBE(n_cells, n_nodes, nv)
        _, isonb = tge.get_BOUNDARY(n_cells, n_nodes, nbe, nv)
        max_elements = int(np.bincount(nv.ravel(), minlength=n_nodes).max(initial=0))
        nbve, nbvt, ntve = tge.get_NBVE_NBVT(n_nodes, n_cells, nv, max_elements)
        tge_ntsn, tge_nbsn, _, _, invalid = tge.get_NTSN_NBSN(
            nbve, ntve, nbvt, nbe, nv, isonb, max_elements, n_nodes
        )
        if np.any(invalid):
            indices = np.flatnonzero(invalid)
            raise ValueError(f'TGE found invalid boundary topology at {len(indices)} nodes, e.g. {indices[:10].tolist()}')

        # TGE stores a closed ring, not the unique neighbors needed by smoothing.
        # Interior: [neighbors..., first_neighbor]; boundary: [self, neighbors..., self].
        boundary = isonb != 0
        counts = tge_ntsn - np.where(boundary, 2, 1)
        if np.any(counts <= 0):
            raise ValueError('Some mesh nodes have no valid neighbors for smoothing')
        neighbors = np.full((n_nodes, int(counts.max(initial=0))), -1, dtype=np.int64)
        for node in range(n_nodes):
            first = 1 if boundary[node] else 0
            count = int(counts[node])
            neighbors[node, :count] = tge_nbsn[node, first:first + count]

        self.nbsn, self.ntsn = neighbors, counts
        # Preserve the existing public edges attribute and its Matplotlib ordering.
        self.edges = Triangulation(self.x, self.y, self.tri).edges