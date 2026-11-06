# fvtools - tools to interact with FVCOM data
a variety of scripts to interact with FVCOM data before, during and after a model run.

# Setup
Here, we demonstrate how to install `fvtools` on Olivia and how to run the fvcom model setup scripts interactively from a compute node on that machine.

First, we clone fvtools to your home directory
```bash
cd
git clone git@github.com:akvaplan-niva/fvtools.git
mkdir container
cd container
```

Sigma2 wants you to install python packages using the container wrapper on Olivia, here we'll assume that you will install it from a container catalog
```bash
module load NRIS/CPU hpc-container-wrapper
conda-containerize new --mamba --prefix . ~/fvtools/env.yml
```

You now need to add the container to your PATH and add alises to activate the container to your ~/.bashrc
```bash
alias vnode='salloc --ntasks=3 --cpus-per-task=1 --time=02:00:00 --mem-per-cpu=2G --qos=devel --account=NN9238K'
alias pton='module load NRIS/CPU hpc-container-wrapper && export PATH="/cluster/home/${USER}/container/bin:$PATH" && export http_proxy=http://10.63.2.48:3128/ && export https_proxy=http://10.63.2.48:3128/'
```

- `vnode` will ask for a virtual node where you will use the container
- `pton` will activate the container on the compute node and give the compute node access to the internet

Start a virtual session on a compute node
```bash
vnode
```
and note which node you are allocated.
```bash
hes001@c1-78:~>
```
Start a `jupyter lab` session and note the link to it (not the one with the compute node id)
```bash
pton
jupyter lab --no-browser --port=8080 --ip=0.0.0.0
-->http://127.0.0.1:8080/lab?token=c2d0b67cef8680e4ca44bad6e7ca8cdf054bfc2d5c145c0d
```

forward the 8080 port to the computer you're working on
```bash
PS C:\Users\hes> ssh -L 8080:c1-78:8080 hes001@olivia.sigma2.no
```
you should now be able to run `fvtools` from a jupyter lab running on the cluster remotely from the browser on your personal computer.

# General idea
These scrips were developed to emulate the workflow from `fvcom_toolbox/fvtools` in MATLAB

`We always`:
- Prepare our mesh in SMS, where we write a `.2dm` file with a nodestring on the outer boundaries
- Use high-resolution bathymetry datasets downloaded from Kartverket with approval from the Norwegian Navy.
- Use atmospheric forcing from the AROME configuration "MetCoOp" which is accessible on thredds.met.no

`We most of the time`:
- Work in Norway, where input data are most easily accessible in `UTM33W` coordinates
- Nest into existing `FVCOM` experiments using files stored on `Betzy` or on `Stokes`

`We sometimes`:
- Nest into larger domain ROMS models operated by the met office or the institute of marine research
  - `NorFjords160` is a series of 160 m resolution ROMS models configured to model fjords.
  - `NorKyst800` is an 800 m ROMS model configured for near-coast modelling.
  - `NorShelf2.5km` is a 2500 m resolution data assimilated ROMS model configured for shelf modelling.

You can call most scripts via the `main` function, which will create input/forcing files following a standardized setup:
- `fvtools.setup.BuildCase.main` 
  - Quality controls the mesh (not complete; some rare cases, such as nodes connected to two boundaries, will not be detected yet)
  - Smooths raw bathymetry to desired `rx0` value (i.e. following MetOffice ROMS routines)
  - Returns estimate of necessary CFL criteria
  - Creates FVCOM.dat input files
    - `casename_cor.dat`
    - `casename_grd.dat`
    - `casename_obc.dat`
    - `casename_spg.dat`

- `fvtools.setup.BuildRivers.main` creates river forcing for your experiment
- `fvtools.setup.nesting.get_ngrd.main` cuts out a nesting zone mesh from the main mesh
  - for fvcom to fvcom nested domains, this routine will dump the `casename_bathymetry.dat` file to your `input` folder 
- `atmospheric forcing: fvtools.setup.atm.read_metCoop.main`
  - interpolates atmospheric forcing to your domain

Boundary forcing is created by these modules:
- `fvtools.setup.nesting.roms.main` (ROMS / NorKyst)
- `fvtools.setup.nesting.fvcom.main` (FVCOM mother)
- `fvtools.setup.nesting.cmems.main` (CMEMS/GLORYS)
- `fvtools.setup.nesting.tpxo.main` (add TPXO tides to a nesting file)

they will produce `.nc` forcing files for the boundary and `.dat` depth files, where the depth near the open boundaries is set equal to that of the forcing model, with a relaxation zone connecting it to the bathymetry set by BuildCase.

# Workflow:
A typical Akvaplan FVCOM experiment is made following these steps:

## Preparing an experiment
1. Create a folder called `casename` and a subfolder called `input`.
2. Put a `casename_sigma.dat` file into `casename/input`, for example, a TANH sigma coordinate:
```dat
NUMBER OF SIGMA LEVELS = 35
SIGMA COORDINATE TYPE = TANH
DU = 2.5
DL = 0.5
```

### Preparing the mesh
#### Writing .dat grid files
You have a `casename.2dm` file (either from smeshing or from SMS); create the FVCOM grid input files using `BuildCase`:
```python
import fvtools.setup.BuildCase as bc
bc.main('cases/inlet/casename.2dm', 'bathymetry.txt')
```
`Mesh quality`: BuildCase prepares the grid before a simulation. It checks if there are invalid triangles (with more than one side facing the boundary) and will remove such triangles if present.

`Bathymetry`: BuildCase smooths the bathymetry to a  [rx0](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwil_uzfvL39AhWPnYsKHZREAtAQFnoECAUQAQ&url=http%3A%2F%2Fmathieudutour.altervista.org%2FPresentations%2FSteepnessPresExt.pdf&usg=AOvVaw1bBZEBsmkvWmSeCkxlrE3y) factor less than `rx0max`. 

`Boundary setup`: It finds the `OBC` nodes, the `sponge nodes` and sets a `sponge factor` if asked to (defaults to 0).

`BuildCase` returns `casename_*.dat` FVCOM input files to the `input` folder and a file called `M.npy` used by other setup routines.

#### The mesh in the nesting zone
fvtools supports ROMS, FVCOM and CMEMS/GLORYS nesting (using the same `ngrd.npy` nesting grid):
- From `ROMS` (`NorKyst`, `NorShelf`, or IMR NorFjords)
- From `FVCOM` (from any `FVCOM` model that overlaps with this mesh)
- From `CMEMS/GLORYS` (local combined NetCDF)

```python
import fvtools.setup.nesting.get_ngrd as gn

# ROMS-FVCOM nesting:
# nrows is the number of "square" rows from the obc used for the nestingzone. We also normally remove land squares in the nestingzone on all rows except
# the row that is in contact with the OBC.
gn.main('M.npy', nrows = 4, remove_land_squares = True)

# FVCOM-FVCOM nesting:
gn.main('M.npy', mother='mother_fvcom.nc')

# These routines will create an ngrd.npy file in your working directory
```

### Nesting (OBC forcing)
Interpolating ocean state to the FVCOM nesting zone.

`FVCOM to FVCOM nesting` requires a [filelist](#a-filelist-linking-to-fvcom-results).
```python
import fvtools.setup.nesting.fvcom as f2f
f2f.main(ngrd = 'ngrd.npy', 
         fileList = 'fileList.txt', 
         output_file = '/input/my_experiment_nest.nc', 
         start_time = '2018-01-01-00', 
         stop_time = '2018-02-01-00')

```

`ROMS-FVCOM nesting` will automatically find the forcing it needs at thredds.met.no:
```python
import fvtools.setup.nesting.roms as rn
rn.main('M.npy', 'ngrd.npy', './input/casename_nest.nc', '2018-01-01-00', '2018-02-01-00', mother='H-NS')
```

MET Norway does _not_ guarantee data is continuous in time. We must keep the tides continuous; for that, we use `fvtools.setup.nesting.fill_nesting_gaps`. It performs a tidal analysis for `zeta` at every `node`, `ua, va` at every `cell`, and `u, v` in every `sigma layer` at every `cell` to fill gaps. Temperature and salinity are interpolated linearly in time. See `fvtools.setup.nesting.fill_nesting_gaps` for the gap-filling implementation.

`fvtools.setup.nesting.roms` supports these model identifiers (`mother=`):
- `NKv2` — NorKyst v2
- `NKv3` — operational NorKyst v3 forecast (staggered ROMS C-grid)
- `NKv3-hindcast` or `NKv3H` — NorKyst v3 hindcast (collocated lon/lat grid)
- `H-NS` and `D-NS` — hourly/daily NorShelf
- `IMR-NF` — IMR NorFjords

Adding a ROMS reader to `fvtools.grid.roms_grid` can support other ROMS experiments.

#### CMEMS/GLORYS and TPXO nesting

CMEMS nesting starts from a **local combined GLORYS NetCDF** containing `thetao`, `so`, `uo`, `vo`, `zos` and static `deptho`. The download helper can create this combined file using the Copernicus Marine service (requires access to its data):

```python
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.setup.nesting.cmems_download import download_cmems
import fvtools.setup.nesting.cmems as cmems_nest

M = FVCOM_grid('M.npy')
download_cmems(M, '2024-04-01', '2024-04-30', outfile='input/cmems/glorys.nc')
cmems_nest.main('M.npy', 'ngrd.npy', 'input/cmems/glorys.nc',
                'input/casename_nest.nc', latlon=True)
```

CMEMS/GLORYS supplies temperature, salinity, sea-surface elevation and currents on its native daily timestamps. The routine matches nesting-zone bathymetry and computes horizontal/vertical interpolation without resampling time; FVCOM handles time interpolation. Run it using an original, **unmatched** FVCOM `M.npy` when changing the mother model.

Add TPXO tides **after** CMEMS nesting (do not add the same tides twice):

```python
import fvtools.setup.nesting.tpxo as tpxo_nest

tpxo_nest.main(
    'input/casename_nest.nc',
    harmonics_files={'zeta': 'tpxo/h.nc', 'u': 'tpxo/u.nc', 'v': 'tpxo/v.nc'},
    tpxo_bathymetry='tpxo/grid.nc',
)
```

The TPXO U/V harmonics are transports and require TPXO `hu`/`hv` bathymetry to convert them to velocities. For elevation-only tides, pass only `{'zeta': 'tpxo/h.nc'}` and omit `tpxo_bathymetry`. **Projected FVCOM nesting with TPXO currents is not yet supported** by `nesting.tpxo`; use a spherical (lat/lon) setup for that combination.

### River runoff
The river runoff is given for geographical catchment areas ids (vassdrag), and the ids are mapped at [nve atlas](https://atlas.nve.no/).
```python
import fvtools.setup.BuildRivers as br
await br.main('2018-01-01', '2018-02-01', vassdrag, 'M.npy')
```
The routine writes a file called `RiverNamelist.nml` and `riverdata.nc` to your working directory. Put these in the `input` folder.

### Atmospheric forcing
We use the MetCoOp-AROME model for atmospheric forcing.
```python
import fvtools.setup.atm.read_metCoop as rm
rm.main("M.npy", "./input/casename_atm.nc", "2018-01-01-00", "2018-02-01-00")
```

### Initial conditions
#### fvtools.setup.initial.fvcom
It interpolates initial fields from an FVCOM mother to a restart file.
- using an existing `restartfile.nc`
- make a restartfile for a `startdate`

```python
import fvtools.setup.initial.fvcom as ir
ir.main(child_fn="casename_restart_0001.nc", 
        filelist="filelist_restart.txt")

# alternatively, if you need to make a restart file
ir.main(startdate="2023-01-31-00",
        child_grid="M.npy",
        filelist="filelist_restart.txt")

# If you don't have a filelist
ir.main(child_fn="casename_restart_0001.nc", 
        result_folder="./mother_folder/output01/", 
        name = "mother_restart")
```

#### fvtools.setup.initial.roms
Interpolate initial fields from a ROMS model to your restart file, but be warned: the model may crash if you set `uv=True`.
```python
import fvtools.setup.initial.roms as ir
ir.main(restartfile="casename_restart_0001.nc", mother="H-NS", uv=True)
```

#### CMEMS/GLORYS restart, followed by TPXO

`fvtools.setup.initial.cmems` reads a **local combined GLORYS NetCDF** (including `deptho`). It updates an existing FVCOM restart or creates a new one with `restart_date`. The source provides `zeta`, temperature and salinity and, with `uv=True`, cell-centred `u/v/ua/va`. Use `time_method='nearest'` explicitly when the restart time differs from the native daily CMEMS timestamp.

```python
import fvtools.setup.initial.cmems as cmems_restart

cmems_restart.main(
    restartfile='input/casename_restart.nc',
    restart_date='2024-04-18-12',  # omit for an existing restart file
    fvcom_grid='M.npy',
    glorys_file='input/cmems/glorys.nc',
    uv=True,
    latlon=True,
    time_method='nearest',
    max_time_difference_hours=12,
)
```

Add TPXO to the resulting restart **after** interpolating CMEMS. Tide elevation is added to `zeta`; optional U/V currents are depth-uniform and are also added to `ua/va`. The routine is atomic and rejects applying TPXO twice.

```python
import fvtools.setup.initial.tpxo as tpxo_restart

tpxo_restart.main(
    'input/casename_restart.nc',
    harmonics_files={'zeta': 'tpxo/h.nc', 'u': 'tpxo/u.nc', 'v': 'tpxo/v.nc'},
    tpxo_bathymetry='tpxo/grid.nc',
)
```

For elevation-only TPXO, provide only `zeta`. If the restart has no `glorys_velocity_coordinates` metadata and you add currents, specify `latlon=True` or `False` explicitly. See [examples/README.md](examples/README.md) for the current import paths.

## After an experiment
### A filelist linking to FVCOM results
You have now run FVCOM, which has stored results to output folders, e.g. output1, output2, output3. Store paths pointing to these folders in a "folders.txt" file:
```txt
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output01
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output02
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output03
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output04
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output05
```
fvcom_make_file_list.py makes a file that links points in time to files and indices in FVCOM results.
```bash
python -m fvtools.util.fvcom_make_file_list -d folders.txt -s PO10 -f fileList.txt
```

### Take a quick look at the results
`qc_gif` and `qc_gif_uv` create animations of scalar or velocity fields between `[start, stop]` (or the entire timespan if not specified).

They help look for dynamically active regions so that you can avoid putting a nesting zone there. (we want to put the nesting zone in areas where the flow fluctuates on timescales >> 1h to avoid aliasing and false shocks).

#### qc_gif
Plots any field stored on nodes, either as a sigma-layer, z-level, or transect movie. It accepts a filename, a folder or a filelist as input.
```python
import fvtools.plot.qc_gif as qc
qc.main(folder='output01', var=['salinity', 'temp'], sigma=0)
qc.main(filelist='filelist.txt', var=['salinity', 'temp'], z=10)

# transect movie (from file by setting setting=file.txt, or graphically by setting section = True)
qc.main(fname='output01/casename.nc', var=['salinity', 'temp', 'tracer_01'], section='transect.txt')
qc.main(folder='output01', var=['salinity', 'temp'], section=True)
```

#### qc_gif_uv
Creates quick georeferenced gifs of velocity fields. It accepts `filelist` and `folder`, `sigma` and `z`.
```python
import fvtools.plot.qc_gif_uv as qc
qc.main(folder='output01', sigma=0)
qc.main(folder='output01', z=10)

```
## The mesh object - FVCOM_grid
The mesh object is the main interface for looking at an FVCOM grid. It reads a variety of input formats:
- `casename_xxxx.nc`
- `casename_restart_xxxx.nc`
- `M.npy`
- `M.mat`
- `casename.2dm`
- `smeshing.txt`

It provides simple functions to look at a mesh and results. Methods to export the mesh, write `*.dat` input files etc.

`Example use:`
```python
from fvtools.grid.fvcom_grd import FVCOM_grid
M = FVCOM_grid("casename_0001.nc")

# Get a quick summary of all attributes and functions:
M?

# See the mesh
M.plot_grid()

# Write the mesh to a .2dm file so that you can edit it in SMS
M.write_2dm()

```

You can georeference data
```python
M.georeference()
M.plot_grid()
```

### Plot results
Data stored on nodes can be plotted on node-based control volume patches, for example:
```python
M.georeference()
M.plot_cvs(M.h)
```

Alternatively plot on triangle patches (much faster the first time):
```python
M.plot_field(M.h)

```

### Check mass conservation
```python
tracer = M.load_netCDF('casename_0001.nc', 'fabm_tracer')
for i in range(len(tracer[:,0,0])):
  mass = np.sum(M.node_volume * tracer[i,:].T)
  plt.scatter(i, mass)

```

### Interpolate data to z-levels
```python
# Get salinity at 50 m depth and look at it
salt_50m = M.load_netCDF('casename_0001.nc', 'salinity', 0, z=50)
M.georeference()
M.plot_contour(salt_50m)
```
