"""FVCOM mesh loading, direct initialization and sigma configuration.

Private implementation module. Public GridLoader remains available
from fvtools.grid.fvcom_grd.
"""

import numpy as np
from netCDF4 import Dataset

from fvtools.grid.depth import setup_sigma


class GridLoader:
    '''
    Loads relevant grid data to be used as instance attributes
    '''
    def _direct_initialization(self, x = None, y = None, tri = None,
                               lon = None, lat = None,
                               h = None, siglay = None, siglev = None,
                               obc_nodes = None, cropped_nodes = np.array(0), cropped_cells = np.array(0)):
        '''
        Initialize a new FVCOM_grid object using known fields
        '''
        # Assertions to make sure that input is consistent, regardless of source
        assert tri.min() == 0, 'Triangulation is invalid'
        if obc_nodes is None:
            obc_nodes = []
        assert isinstance(obc_nodes, list), 'obc_nodes must be given as a list'
        assert type(cropped_nodes) == np.ndarray, 'cropped nodes must be a numpy array'
        assert type(cropped_cells) == np.ndarray, 'cropped cells must be a numpy array'
        if h is not None: assert h.min() > 0, 'Minimum depth is invalid'

        # Add more tests when bugs appear
        self.tri = tri
        if x is None and y is None:
            self.x, self.y = np.zeros(tri.max()+1), np.zeros(tri.max()+1)
        else:
            self.x, self.y = np.squeeze(x), np.squeeze(y)

        if lon is None and lat is None:
            self.lon, self.lat = np.zeros(self.x.shape), np.zeros(self.y.shape)
        else:
            self.lon, self.lat = lon, lat

        if h is not None:   
            self.h = np.squeeze(h)
        else:
            self.h = h

        self.siglay, self.siglev = siglay, siglev

        # Identifiers
        self.obc_nodes, self.cropped_nodes, self.cropped_cells = obc_nodes, cropped_nodes, cropped_cells

        assert self.x.shape == self.y.shape, 'x, y shapes do not match'
        assert len(self.x.shape) == 1 and len(self.y.shape) == 1, 'shape of x and y needs to be (N,)'
        assert self.lon.shape == self.lat.shape, 'lon, lat shapes do not match'

    def _add_grid_parameters_2dm(self):
        '''
        Grid parameters from a .2dm file
        - assumes that .2dm files are always projected to carthesian coordinates
        '''
        import fvtools.grid.fvgrid as fvgrid
        tri, nodes, x, y, _, _, types = fvgrid.read_sms_mesh(self.filepath, nodestrings = True)
        l = []
        for nstr in types:
            l.extend(nstr)
        self._direct_initialization(x=x, y=y, tri=tri, obc_nodes = l)

        # Add nodestrings
        nodestrings = []
        for ns in types:
            nodestrings.append(np.array(ns))
        self.nodestrings = nodestrings
        self.casename = self.filepath.split('.2dm')[0]

    def _add_grid_parameters_mat(self):
        '''
        Read grid attributes from matlab grid file and add them to the FVCOM_grid instance
        - Probably won't work that well anymore
        '''
        from scipy.io import loadmat
        grid_mfile = loadmat(self.filepath)
        self.casename = 'from_mat_file'
        self._direct_initialization(
            x = grid_mfile['Mobj']['x'][0][0][:,0], 
            y = grid_mfile['Mobj']['y'][0][0][:,0],
            tri = grid_mfile['Mobj']['tri'][0][0][:]-1, 
            h = grid_mfile['Mobj']['h'][0][0][:,0],
            siglay = grid_mfile['Mobj']['siglay'][0][0], 
            siglev = grid_mfile['Mobj']['siglev'][0][0], 
            obc_nodes = list(grid_mfile['Mobj']['obc_nodes'][0][0][0]-1),
            lon = grid_mfile['Mobj']['lon'][0][0][:,0],
            lat = grid_mfile['Mobj']['lat'][0][0][:,0]
            )

    def _add_grid_parameters_nc(self):
        """
        Load ncdata from a fvcom-formated netCDF file
        """
        kwargs = {}
        # Add casename
        self.casename = self.filepath.split('/')[-1].split(self.filepath.split('/')[-1].split('_')[-1])[0][:-1]
        if 'restart' in self.filepath.split('/')[-1]:
            self.casename = self.filepath.split('/')[-1].split('_restart')[0]

        with Dataset(self.filepath) as d:
            obc_nodes = []
            if 'obc_nodes' in d.variables.keys():
                kwargs['obc_nodes'] = list(d['obc_nodes'][:]-1) # fortran to python indexing
            kwargs['x'], kwargs['y'], kwargs['lon'], kwargs['lat'], kwargs['tri'] = d['x'][:], d['y'][:], d['lon'][:], d['lat'][:], d['nv'][:].T-1
            kwargs['h'] = d['h'][:]
            if 'siglay' in d.dimensions.keys():
                kwargs['siglev'], kwargs['siglay'] = d['siglev'][:].T, d['siglay'][:].T
        self._direct_initialization(**kwargs)

    def _add_grid_parameters_npy(self):
        '''
        Load grid information stored in a M.npy file
        '''
        grid = np.load(self.filepath, allow_pickle=True).item()
        obc_nodes = []
        if 'obc_nodes' not in list(grid.keys()):
            obcs = [list(nodestring[0]) for nodestring in grid['read_obc_nodes'][0]]
            for obc in obcs:
                obc_nodes.extend(obc)
        else:
            obc_nodes = list(grid['obc_nodes'])

        self._direct_initialization(x = grid['x'], y = grid['y'], 
                                    lon = grid['lon'], lat = grid['lat'],
                                    tri = grid['nv'], h = grid['h'], 
                                    siglay = grid['siglay'], siglev = grid['siglev'], 
                                    obc_nodes = obc_nodes)

        # Add nodestrings (for convenience, computing them can be quite slow on huge grids)
        if 'read_obc_nodes' in grid.keys():
            self.nodestrings = [list(nodestring[0]) for nodestring in grid['read_obc_nodes'][0]]
        elif 'nodestrings' in grid.keys():
            self.nodestrings = grid['nodestrings']

        if 'info' in grid.keys():
            self.info = grid['info']
            self.casename = self.info['casename']
            try:
                self.reference = self.info['reference']
            except:
                pass
        else:
            self.casename = 'FVCOM experiment'

    def _add_grid_parameters_txt(self):
        '''Grid parameters from smeshing file'''
        with open(self.filepath) as f:
            nodenum = int(f.readline())
            points  = np.loadtxt(f, delimiter = ' ', max_rows = nodenum)
            trinum  = int(f.readline())
            tri     = np.loadtxt(f, delimiter = ' ', max_rows = trinum, dtype=int)
        self._direct_initialization(x = points[:,0], y = points[:,1], tri = tri)

    def read_sigma(self, sigmafile=None, **sigma_kwargs):
        """
        Read FVCOM sigma settings and calculate nodal sigma coordinates.

        Parameters
        ----------
        sigmafile : str or pathlib.Path, optional
            Path to a *_sigma.dat file.
        **sigma_kwargs
            Optional overrides passed to depth.setup_sigma.

        Notes
        -----
        KB is the number of vertical interfaces.
        KBM1 = KB - 1 is the number of layers.
        Both siglev and siglay run from surface to bottom.
        """
        if sigmafile is None:
            sigmafile = f"input/{self.casename}_sigma.dat"

        kb, stype, parameters = self._read_sigma_configuration(sigmafile)
        options = self._sigma_options(stype, parameters)
        options.update(sigma_kwargs)

        # Only Souza and generalized coordinates depend on bathymetry.
        depth = getattr(self, "_h", None)

        if depth is None:
            if stype in ("souza", "generalized"):
                raise ValueError(f"{stype.upper()} sigma coordinates require nodal bathymetry (h)")
            depth = np.ones(len(self.x))

        depth = np.atleast_1d(np.asarray(depth))

        if depth.size != len(self.x):
            raise ValueError("The number of depth values must match the number of grid nodes")

        # setup_sigma expects KBM1, not KB.
        result = setup_sigma(depth=depth, n_layers=kb - 1, stype=stype, **options)

        self.siglev = result["z"]
        self.siglay = result["zz"][:, :kb - 1]

    @staticmethod
    def _read_sigma_configuration(sigmafile):
        """
        Parse an FVCOM sigma configuration file.
        """
        import re

        parameters = {}
        values = []

        with open(sigmafile, encoding="utf-8") as file:
            for line in file:
                line = line.split("!", 1)[0].split("#", 1)[0].strip()

                if not line or "=" not in line:
                    continue

                name, value = line.split("=", 1)
                name = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
                value = value.strip().strip("\"'")

                parameters[name] = value
                values.append(value)

        if len(values) < 2:
            raise ValueError(f"Invalid FVCOM sigma file: {sigmafile}")

        # Support positional parameters from older sigma files.
        parameters["_ORDERED_VALUES"] = values

        kb_value = parameters.get("NUMBER_OF_SIGMA_LEVELS", parameters.get("KB", values[0]))

        stype = parameters.get("SIGMA_COORDINATE_TYPE", parameters.get("STYPE", values[1])).strip().lower()

        try:
            kb = int(kb_value)
        except ValueError as exc:
            raise ValueError(f"Invalid number of sigma levels (KB): {kb_value}") from exc

        if kb < 3:
            raise ValueError("KB must be at least 3")

        return kb, stype, parameters

    @staticmethod
    def _sigma_options(stype, parameters):
        """Translate FVCOM sigma parameters to setup_sigma arguments."""

        def value(*names, position=None, required=True, default=None):
            for name in names:
                if name in parameters:
                    return float(parameters[name])

            values = parameters.get("_ORDERED_VALUES", [])

            if position is not None and len(values) > position:
                return float(values[position])

            if required:
                raise ValueError(f"{stype.upper()} sigma requires one of: {', '.join(names)}")

            return default

        if stype == "uniform":
            return {"p_sigma": value("P_SIGMA", position=2, required=False, default=1.0)}

        if stype == "geometric":
            return {"p_sigma": value("P_SIGMA", position=2)}

        if stype == "tanh":
            return {"du2": value("DU", "DU2", position=2), "dl2": value("DL", "DL2", position=3)}

        if stype == "souza":
            return {
                "tcline": value("TCLINE", "TCLIN", position=2),
                "theta_s": value("THETA_S", "THETAS", position=3),
                "theta_b": value("THETA_B", "THETAB", position=4)
            }

        if stype == "generalized":
            options = {}

            for field in ("hmin1", "duu", "dll", "ku", "kl"):
                if field.upper() in parameters:
                    options[field] = value(field.upper())

            for field in ("ku", "kl"):
                if field in options:
                    options[field] = int(options[field])

            for field in ("zku", "zkl"):
                if field.upper() in parameters:
                    raw = parameters[field.upper()].replace(",", " ")
                    options[field] = [
                        float(x) for x in raw.split()
                    ]

            return options

        raise ValueError(f"Unsupported sigma coordinate type: {stype!r}")
