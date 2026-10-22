import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.plot.movie_common import (
    get_movie_frames, print_frame_range, save_animation,
    make_horizontal_figure, make_section_figure, get_grid_georeference,
)

# Synthetic section fields (not variable names in the FVCOM NetCDF file).
SECTION_VELOCITY_FIELDS = {
    'normal_velocity': 'normal',
    'along_velocity': 'along',
    'velocity_speed': 'speed',
}

# Only the speed magnitude is cell-centred. Vorticity and PV are nodal.
CELL_VELOCITY_FIELDS = frozenset({'sp'})
    
# ----------------------------------------------------------------------------------------
#                     Create an animation from a number of files
# ----------------------------------------------------------------------------------------
def main(folder=None, fname=None, filelist=None, var=None,
         sigma=None, z=None, xlim=None, ylim=None, start=None, stop=None,
         section=None, section_res=None, fps=12, mname=None, dpi=100,
         nlevels=20, reference='epsg:32633', **kwargs):
    """Create FVCOM animations from sigma, z-level, and/or vertical sections.

    For section normal-flow animations, specify section=... and
    var=['normal_velocity']; optional section_vlim is in m/s.
    """
    options = _prepare_movie_options(
        folder=folder, fname=fname, filelist=filelist, var=var,
        sigma=sigma, z=z, xlim=xlim, ylim=ylim, start=start, stop=stop,
        section=section, section_res=section_res, fps=fps, mname=mname,
        dpi=dpi, nlevels=nlevels, reference=reference, **kwargs,
    )

    if options['sigma'] is not None:
        surface_movie(**options)
    if options['z'] is not None:
        zlevel_movie(**options)
    if options['section'] not in (None, False):
        section_movie(**options)
    print('--> Done.')


def _prepare_movie_options(folder=None, fname=None, filelist=None, var=None,
              sigma=None, z=None, xlim=None, ylim=None, start=None,
              stop=None, section=None, section_res=None, fps=12,
              mname=None, dpi=100, nlevels=20,
              reference='epsg:32633', **kwargs):
    """Return all options and selected frames for the three movie functions.

    The returned dictionary can be passed directly to surface_movie,
    zlevel_movie or section_movie. Only called internally by main().
    """
    options = dict(
        folder=folder, fname=fname, filelist=filelist,
        var=['salinity', 'temp', 'zeta'] if var is None else (
            [var] if isinstance(var, str) else list(var)),
        sigma=sigma, z=z, xlim=xlim, ylim=ylim, start=start, stop=stop,
        section=section, section_res=section_res, fps=fps, mname=mname,
        dpi=dpi, nlevels=nlevels, reference=reference,
    )
    options.update(kwargs)
    if options['section'] in (None, False) and options['sigma'] is None and options['z'] is None:
        raise ValueError('Specify at least one of section, sigma or z')
    if options['sigma'] is not None and options['z'] is not None:
        raise ValueError('Specify either sigma or z, not both')
    frames = get_movie_frames(
        folder=options['folder'], fname=options['fname'],
        filelist=options['filelist'], start=options['start'],
        stop=options['stop'],
    )
    options['time'], options['dates'] = frames.time, frames.datetime
    options['List'], options['index'] = frames.path, frames.index
    options['cb'] = build_colorbars(
        frames, options['var'], sigma=options['sigma'],
        nlevels=options['nlevels'],
    )
    print_frame_range(frames)
    return options

    
# ----------------------------------------------------------------------------------------------------------------------
#                                            Movie plotters
# ----------------------------------------------------------------------------------------------------------------------
def surface_movie(**kwargs):
    """Make horizontal contour movies on a specified sigma layer."""
    _validate_scalar_fields(kwargs['var'])
    maker = HorizontalMaker(**kwargs)
    _write_horizontal_movies(
        maker, kwargs, callback=maker.contourf_animate,
        suffix=f'sigma_{kwargs["sigma"]}',
    )


def zlevel_movie(**kwargs):
    """Make horizontal contour movies interpolated to constant z levels."""
    _validate_scalar_fields(kwargs['var'])
    maker = HorizontalMaker(**kwargs)
    maker.sigma = None
    _write_horizontal_movies(
        maker, kwargs, callback=maker.zlevel_animate,
        suffix=f'{kwargs["z"]}m_depth',
        excluded=('zeta', 'vorticity', 'pv', 'sp'),
    )


def _validate_scalar_fields(fields):
    invalid = set(fields).intersection(SECTION_VELOCITY_FIELDS)
    if invalid:
        raise ValueError(f'Section velocity fields require section=...: {sorted(invalid)}')


def _write_horizontal_movies(maker, options, *, callback, suffix, excluded=()):
    """Select fields and hand each figure to the shared animation writer."""
    name = options['mname'] or maker.M.casename
    count = len(options['time'])
    for field in options['var']:
        if field in excluded:
            continue
        maker.var = field
        save_animation(
            make_horizontal_figure(maker.M, dpi=options['dpi']),
            callback,
            frame_count=count,
            output_stem=f'{name}_{suffix}_{field}',
            fps=options['fps'],
            progress_label=f'- Make {field} movie: ',
            maker=maker,
            grid=maker.M,
        )


def _set_section_velocity_colorbars(mmaker, fields, vlim, nlevels):
    """Use one symmetric colour scale across all section frames.

    If no explicit limit is supplied, estimate from first/middle/last
    timestep (a fast approximation, not a guaranteed global maximum).
    """
    vel_fields = [f for f in fields if f in SECTION_VELOCITY_FIELDS]
    if not vel_fields:
        return
    if vlim is not None:
        if not np.isscalar(vlim) or not np.isfinite(vlim) or vlim <= 0:
            raise ValueError('section_vlim must be a positive finite number in m/s')
        limit = float(vlim)
    else:
        sample_indices = np.unique(np.linspace(0, len(mmaker.files)-1, 3, dtype=int))
        samples = []
        with mmaker.M.netcdf_session():
            for j in sample_indices:
                mmaker.M.zeta = mmaker.M.load_netCDF(
                    mmaker.files[j], 'zeta', mmaker.index[j]
                )
                vel = mmaker.M.load_netCDF(
                    mmaker.files[j], ('u', 'v'), mmaker.index[j]
                )
                out = mmaker.M.get_section_velocity(
                    vel['u'], vel['v'],
                    velocity_coordinates=mmaker.velocity_coordinates
                )
                for field in vel_fields:
                    a = np.asarray(out[SECTION_VELOCITY_FIELDS[field]])
                    a = np.abs(a[np.isfinite(a)])
                    if a.size:
                        samples.append(a.max())
        limit = max(float(np.max(samples)), 1e-3) if samples else 0.1
        print(f'- Sampled section velocity colour scale: {limit:.3g} m/s '
              '(first/middle/last frames)')

    for field in vel_fields:
        cfg = mmaker.cb[field]
        cfg['label'] = 'm/s'
        cfg['extend'] = 'both'
        cfg['cmap'] = cmo.cm.speed if field == 'velocity_speed' else cmo.cm.balance
        cfg['colorticks'] = np.linspace(
            0 if field == 'velocity_speed' else -limit,
            limit, max(3, int(nlevels))
        )


def section_movie(**kwargs):
    '''
    Plot movies from cross-sections
    '''
    print('\nCreate the section movie')
    
    # Find grid info
    print('- Get grid info, section points and triangulation')
    M = FVCOM_grid(kwargs['List'][0], verbose = False, reference = kwargs['reference'], static_depth = True)
    section_file = None if kwargs['section'] is True else kwargs['section']

    mname = kwargs['mname'] or M.casename

    print('- Prepare the section')
    M.prepare_section(section_file=section_file, res=kwargs['section_res'], store_transect_img=True)

    # All FVCOM u/v fields are cell-centred: include neighbouring cells
    # across the section, while avoiding a full-domain read per frame.
    n_neighbors = kwargs.get('section_neighbors', 12)
    if (not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 3):
        raise ValueError('section_neighbors must be an integer >= 3')
    n_neighbors = min(int(n_neighbors), M.cell_number)
    points = np.column_stack((M.x_sec, M.y_sec))
    _, closest = M.cell_tree.query(points, k=n_neighbors)
    indices = np.unique(np.asarray(closest).ravel())
    print('- Subgrid to deal with smaller data amounts')
    C = M.subgrid(cells=indices)

    # Reuse the EXACT same sampled positions, without resampling them.
    C.x_sec, C.y_sec = M.x_sec.copy(), M.y_sec.copy()
    section = C.get_section_data(C.h)

    # Create the movie maker
    print('- Prepare the movie maker')
    mmaker   = VerticalMaker(
        kwargs['time'], kwargs['dates'], kwargs['List'], kwargs['index'], 
        kwargs['cb'], ylimit = [section['h'].min()-1, 2]
        )

    # The vertical movie maker only needs to be aware of the cropped grid
    mmaker.M = C
    mmaker.velocity_coordinates = kwargs.get('velocity_coordinates', 'east_north')
    if any(f in SECTION_VELOCITY_FIELDS for f in kwargs['var']):
        _set_section_velocity_colorbars(
            mmaker, kwargs['var'], kwargs.get('section_vlim'),
            kwargs.get('nlevels', 20)
        )

    print('\nMovie maker:')
    for field in kwargs['var']:
        if field == 'zeta':
            continue
        mmaker.var = field
        save_animation(
            make_section_figure(dpi=kwargs['dpi']),
            mmaker.vertical_animate,
            frame_count=len(kwargs['time']),
            output_stem=f'{mname}_section_{field}',
            fps=kwargs['fps'],
            progress_label=f'- Make {field} movie: ',
            maker=mmaker,
            grid=mmaker.M,
        )


#                                  Helper functions
# --------------------------------------------------------------------------------------------------

def initialize_cb(filepath, fields):
    """Initialize colourbar metadata without scanning field values."""
    cb = {}
    with Dataset(filepath) as dataset:
        for field in fields:
            if field in SECTION_VELOCITY_FIELDS:
                units = 'm/s'
            elif field in ('pv', 'vorticity', 'sp'):
                units = ''
            else:
                units = getattr(dataset[field], 'units', '')
            cb[field] = {'min': np.inf, 'max': -np.inf, 'units': units}
    return cb


def setup_colorbar(cb, var=None, nlevels=20):
    """Style colourbars after scanning the selected frames."""
    nlevels = max(3, int(nlevels))
    for field, settings in cb.items():
        lo, hi = settings['min'], settings['max']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = 0.0, 1.0
        if lo == hi:
            padding = max(abs(float(lo))*0.01, 1e-6)
            lo, hi = lo-padding, hi+padding
        settings.update(
            min=lo, max=hi, cmap=cmo.cm.turbid,
            label=settings['units'], norm=None, extend='both',
            colorticks=np.linspace(lo, hi, nlevels),
        )
        if field == 'salinity':
            settings['cmap'] = cmo.cm.haline
        elif field == 'temp':
            settings['cmap'] = cmo.cm.thermal
            settings['label'] = r'$^\circ$C'
        elif field == 'zeta':
            settings['cmap'] = (
                cmo.tools.crop(cmo.cm.balance, lo, hi, 0)
                if lo < 0 < hi else cmo.cm.deep
            )
        elif 'tracer' in field:
            settings['cmap'] = cmo.cm.dense
        elif field == 'vorticity':
            settings.update(cmap=cmo.cm.curl, label='1/s',
                            colorticks=np.linspace(-1e-4, 1e-4, nlevels))
        elif field == 'pv':
            settings.update(cmap=cmo.cm.curl, label='1/ms',
                            colorticks=np.linspace(-1e-5, 1e-5, nlevels))
        elif field == 'sp':
            settings.update(cmap=cmo.cm.speed, label='m/s',
                            colorticks=np.linspace(0, 0.4, nlevels))
    return cb


def _update_field_limits(settings, data):
    """Ignore masked/invalid values when calculating plotting limits."""
    values = np.ma.masked_invalid(np.ma.asarray(data))
    if values.count():
        settings['min'] = min(settings['min'], float(values.min()))
        settings['max'] = max(settings['max'], float(values.max()))


def build_colorbars(frames, fields, sigma=None, nlevels=20):
    """Estimate limits for selected frames, independently of file discovery.

    For z=... (or full 3-D sections), source sigma values determine the
    scale; this may be wider than the actual interpolated field.
    """
    cb = initialize_cb(frames.path[0], fields)
    direct = [f for f in fields if f not in (
        'pv', 'vorticity', 'sp', *SECTION_VELOCITY_FIELDS)]
    if direct:
        for path, indices in frames.iter_files():
            with Dataset(path) as dataset:
                for field in direct:
                    variable = dataset[field]
                    dimensions = variable.dimensions
                    time_axis = dimensions.index('time') if 'time' in dimensions else None
                    vertical_axis = next((i for i, dim in enumerate(dimensions)
                                          if dim in ('siglay', 'siglev')), None)
                    for idx in (indices if time_axis is not None else [None]):
                        selection = [slice(None)] * len(dimensions)
                        if time_axis is not None:
                            selection[time_axis] = idx
                        if vertical_axis is not None and isinstance(sigma, (int, np.integer)):
                            selection[vertical_axis] = sigma
                        _update_field_limits(cb[field], variable[tuple(selection)])
    return setup_colorbar(cb, nlevels=nlevels)



#                                                    Helper classes
# ----------------------------------------------------------------------------------------------------------------------
class AnimationFields:
    '''Fields loaded from FVCOM NetCDF or derived from velocity components.'''
    @property
    def field(self):
        """Load a scalar field with subgrid, sigma/z and wet/dry handling."""
        if self.var in ['pv', 'vorticity', 'sp']:
            return getattr(self, self.var)
        return self.M.load_netCDF(
            self.files[self.i], self.var, self.index[self.i],
            sig=self.sigma, z=getattr(self, 'z', None)
        )

    @property
    def vorticity(self):
        '''
        We can't necessarilly use the cropped grid here since the vorticity calculation at boundaries is fizzy.
        We therefore load velocities for the entire grid when calculating the vorticity.
        '''
        with Dataset(self.files[self.i], 'r') as d:
            _vort = self.T.vorticity(d['ua'][self.index[self.i], :], d['va'][self.index[self.i],:])
        # The TGE vorticity is nodal even though its inputs ua/va are
        # cell-centred. Crop by *nodes*, not by the input velocity cells.
        cropped_nodes = np.asarray(self.M.cropped_nodes)
        if cropped_nodes.ndim == 1 and cropped_nodes.size:
            _vort = _vort[cropped_nodes]
        return _vort

    @property
    def pv(self):
        # Nodal potential vorticity: nodal Coriolis, nodal vorticity,
        # and nodal bathymetric depth. Do not substitute M.hc here.
        return (self.M.f + self.vorticity) / self.M.h

    @property
    def sp(self):
        uv = self.M.load_netCDF(self.files[self.i], ('ua', 'va'),
                                    self.index[self.i])
        return np.hypot(uv['ua'], uv['va'])

class HorizontalMaker(AnimationFields):
    '''
    All the data needed by the 
    '''
    def __init__(
            self, time = None, dates = None, List = None, index = None, var= None, 
            cb = None, xlim = None, ylim = None, reference = None, sigma = None, z = None, **kwargs
            ):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        self.index, self.files, self.time, self.datetime = index, List, time, dates
        self.sigma, self.z = sigma, z
        self.xlim,  self.ylim = xlim, ylim

        # Copy colormap settings
        self.cb = cb

        # Keep the full mesh for vorticity stencils near cropped boundaries.
        full_grid = FVCOM_grid(List[0], reference=reference)
        self.M = full_grid
        if self.xlim is not None and self.ylim is not None:
            self.M = full_grid.subgrid(self.xlim, self.ylim)

        if 'pv' in var:
            self.M.get_coriolis()  # PV lives on nodes, not elements.

        # Only 'sp' is cell-centred among the derived 2-D fields.
        # PV, vorticity, and ordinary scalar fields use nodal extents.
        map_location = 'cell' if set(var).issubset(CELL_VELOCITY_FIELDS) else 'node'
        self.gp = get_grid_georeference(
            self.M, xlim=self.xlim, ylim=self.ylim, at=map_location,
            sources=('hot', 'mapnik', 'voyager'), verbose=True,
        )

        if 'vorticity' in var or 'pv' in var:
            import fvtools.grid.tge as tge
            self.T = tge.main(full_grid, verbose=True)
            self.T.get_art1(verbose=False)

    def contourf_animate(self, i):
        '''
        Write frames of data at z level using tricontourf
        '''
        self.bar.update(i)
        self.i = i
        plt.clf()
        if self.gp is not None:
            plt.imshow(self.gp.img, extent = self.gp.extent)
        if self.var in ['vorticity', 'pv', 'sp']:
            plt.tricontour(
                self.M.x, self.M.y, self.M.tri, self.M.h, 40, colors = 'gray', alpha = 0.5, linewidths = 0.5, zorder = 10
                )
        cont = self.update_figure(field = self.field, title = self.datetime[i].strftime('%d/%B-%Y, %H:%M:%S'))
        return cont

    def zlevel_animate(self, i):
        '''
        Write frames using tricontourf
        '''
        self.bar.update(i)
        self.i = i
        plt.clf()
        if self.gp is not None:
            plt.imshow(self.gp.img, extent = self.gp.extent)
        outdata = self.field
        cont = self.update_figure(
            field = outdata, title = self.datetime[i].strftime('%m/%d, %H:%M:%S') + f' at {self.z} m depth'
            )
        return cont

    def update_figure(self, field=None, title=None):
        '''
        Add cosmetics to the figure
        '''
        if field is not None:
            cont = self.M.plot_contour(
                field, 
                show   = False, 
                cmap   = self.cb[self.var]['cmap'],
                levels = self.cb[self.var]['colorticks'], 
                extend = self.cb[self.var]['extend'], 
                zorder = 5
                )
        if self.xlim is not None and self.ylim is not None:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
        
        if field is not None:
            plt.colorbar(cont, label=self.cb[self.var]['label'], shrink=0.5)

        plt.title(title)   
        if field is not None:
            return cont

class VerticalMaker(AnimationFields):
    '''
    A class that creates cross-section movies
    '''
    def __init__(self, time, dates, List, index, cb, ylimit = None):
        '''
        Let the writer know which frames to make
        '''
        self.cb = cb
        self.index = index
        self.files = List
        self.time  = time
        self.datetime = dates
        self.sigma = None # a bit of a hack
        self.ylimit = ylimit

    def sigma_distance(self, out):
        return np.insert(out['dst'], out['dst'].shape[1], np.nan, axis = 1)
    
    def sigma_depth(self, out):
        return np.insert(out['h'], out['h'].shape[1], np.nan, axis = 1)

    def vertical_animate(self, i):
        '''
        Write a section movie using contourf
        '''
        self.i = i
        self.bar.update(i)
        plt.clf()
        self.M.zeta = self.M.load_netCDF(self.files[i], 'zeta', self.index[i])
        if self.var in SECTION_VELOCITY_FIELDS:
            # Both 3-D velocity components are always defined on FVCOM cells.
            uv = self.M.load_netCDF(
                self.files[i], ('u', 'v'), self.index[i]
            )
            out = self.M.get_section_velocity(
                uv['u'], uv['v'],
                velocity_coordinates=self.velocity_coordinates
            )
            field_data = out[SECTION_VELOCITY_FIELDS[self.var]]
        else:
            out = self.M.get_section_data(self.field)
            field_data = out['transect']

        # Physical section coordinates may be finite even for dry cells;
        # masked values leave those regions blank without shifting the section.
        field_data = np.ma.masked_invalid(field_data)
        valid = field_data.count() > 0 and out['dst'].shape[0] > 1
        if valid:
            cont = plt.contourf(
                out['dst'], out['h'], field_data,
                cmap=self.cb[self.var]['cmap'],
                levels=self.cb[self.var]['colorticks'],
                extend='both'
            )
            plt.colorbar(cont, label=self.cb[self.var]['label'])
        else:
            plt.text(0.5, 0.5, 'No valid data for this time step', ha='center', va='center', transform=plt.gca().transAxes)

        # Plot sigma layers on top to show vertical resolution.
        plt.plot(self.sigma_distance(out), self.sigma_depth(out), c='grey', lw=0.2)

        # Settings
        if self.ylimit is not None:
            plt.ylim(self.ylimit)

        plt.xlabel('km from start of transect')
        plt.ylabel('meter depth')
        plt.title(self.datetime[i].strftime('%m/%d, %H:%M:%S'))
        ax = plt.gca()
        ax.set_facecolor('tab:grey')
