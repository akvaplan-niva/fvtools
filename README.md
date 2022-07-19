# About the workflow in this repository

This repo has two main branches: `master` and `dev`. Both are protected (meaning you have to open a merge request to get stuff in there). `dev` is for active development and is occasionally merged into `master`, which is hopefully more stable and better tested.


This projects is currently under my namespace because it's easier to handle wrt permissions, but we could transfer it to apn once we're up and running I guess.

# fvtools - tools to interact with FVCOM data
a variety of scripts to interact with FVCOM data before, during and after a model run.

[[_TOC_]]


# Example workflow:
## Preparing an experiment
1. Create a folder called `casename` and a subfolder of it called `input`.
2. Put a `casename_sigma.dat` file into `casename/input`, for example a TANH sigma coordinate:
```dat
NUMBER OF SIGMA LEVELS = 35
SIGMA COORDINATE TYPE = TANH
DU = 2.5
DL = 0.5
```

```python
  >>> fvtools = '/home/host/models/fvcom/utils/fvtools/'
  >>> import sys; sys.path.insert(0, fvtools)

```

### Preparing the mesh
#### The full mesh
You have a `casename.2dm` file (either from smeshing or from SMS), and you want to set up a model. The first step is to create the FVCOM grid input files using `BuildCase`:
```python
  >>> import fvtools.pre_pro.BuildCase as bc
  >>> bc.main('cases/inlet/casename.2dm', 'bathymetry.txt')

```
BuildCase returns `casename_*.dat` FVCON input files to the `input` folder, and a file called `M.npy` to be used as input to the rest of the setup routines.

#### The mesh in the nesting zone
fvtools support two nesting types:
- From ROMS (either NorKyst or NorShelf)
- From FVCOM (from any FVCOM model that overlaps with this mesh)

```python
  >>> import fvtools.nesting.get_ngrd as gn

  >>> # ROMS-FVCOM nesting:
  >>> # You need to define the width of the nesting zone (R measured in meters). This is typically approximately 4.5 times the mesh resolution at the OBC.
  >>> gn.main('M.npy', R=5000)

  >>> # FVCOM-FVCOM nesting:
  >>> # The nest nest grid is computed automatically, give the new grid and a path leading to the mother model (used as OBC conditions)
  >>> gn.main('M.npy', mother='mother_fvcom.nc')

```
These routines return a nest-grid file called `ngrd.npy` that you feed to the nesting zone interpolator.

### Nesting (OBC forcing)
Interpolating ocean state to the FVCOM nesting zone

FVCOM to FVCOM nesting requires a [filelist](https://source.coderefinery.org/apn/fvtools/-/blob/hes/README.md#a-filelist-linking-to-fvcom-results) for the mother grid and must be executed from the linux terminal.
```bash
python fvcom2fvcom_nesting.py -n ngrd.npy -f fileList.txt -o ./input/my_experiment_nest.nc -s 2018-01-01-00 -e 2018-02-01-00

```

ROMS-FVCOM nesting will automatically find the forcing it need at thredds.met.no:
```python
  >>> import fvtools.nesting.roms_nesting_fg as rn
  >>> rn.main('M.npy', 'ngrd.npy', './input/casename_nest.nc', '2018-01-01-00', '2018-02-01-00', mother='NS')

```

### River runoff
```python
  >>> import fvtools.pre_pro.BuildRivers as br
  >>> # If you prepare a new experiment (ROMS nested):
  >>> br.main('2018-01-01-00', '2018-02-01-00', temp=None)

  >>> # If you prepare a fvcom-nested experiment:
  >>> br.main('2018-01-01-00', '2018-02-01-00', temp='fvcom_mother_temperatures.npy')

```
This routine writes a file called `RiverNamelist.nml` and `riverdata.nc`. Put these in the `input` folder.

Temperatures for big models are stored on the Stokes and Betzy
  - Stokes: `/data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/`
  - Betzy:  `/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/`


### Atmospheric forcing
We use the MetCoOp-AROME model for atmospheric forcing.
```python
  >>> import fvtools.atm.read_metCoop as rm
  >>> rm.main("M.npy", "./input/casename_atm.nc", "2018-01-01-00", "2018-02-01-00")

```

These are all the input files you need to run a FVCOM experiment (except for `JulianTidesElev`, which may or may not be required for FVCOM-ROMS nesting).

### Initial conditions
Start FVCOM in a coldstart mode using the input files to create a `casename_restart_0001.nc` file. We thereafter interpolate data to it using `interpol_restart` (for fvcom2fvcom experiments) or `interpol_roms_restart` (for roms2fvcom experiments):
#### interpol_restart
Interpolates inital fields from a FVCOM mother to you restartfile. The best practice is to interpolate from a `filelist` referencing to FVCOM restart files (set `-s PO10_restart` when you generate a [filelist](https://source.coderefinery.org/apn/fvtools/-/blob/hes/README.md#a-filelist-linking-to-fvcom-results)), but it is also acceptable to use normal output files - but then with `speed = False`. 
```python
import fvtools.pre_pro.interpol_restart as ir 
ir.main(childfn="casename_restart_0001.nc", filelist="filelist_restart.txt", vinterp=True, speed=True)

# or alternatively avoiding the filelist
ir.main(childfn="casename_restart_0001.nc", result_folder="./mother_folder/output01/", name = "mother_restart", vinterp=True, speed=True)
```

#### interpol_roms_restart
Interpolated initial fields from a ROMS model to your restartfile, but be warned: the model may crash if you set `uv=True`.
```python
import fvtools.pre_pro.interpol_roms_restart as ir 
ir.main("casename_restart_0001.nc", "NS", uv=True, avg=True)
```

## After an experiment
### A filelist linking to FVCOM results
You have now run FVCOM, and it has stored results to output folders, e.g. output1, output2, output3. Store paths pointing to these folders in a "folders.txt" file:
```txt
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output01
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output02
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output03
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output04
/cluster/shared/NS9067K/apn_backup/FVCOM/MATNOC/PO10/output05
```

fvcom_make_file_list.py makes a file that links points in time to files and indices in FVCOM results.
```python
  >>> python fvcom_make_filelist.py -d folders.txt -s PO10 -f fileList.txt

```

### Take a quick look at the results
qc_gif and qc_gif_uv are two versatile scripts to look at your results between `[start, stop]` (or the entire timespan if not specified).

These scripts were developed to be used during a simulation to make it easier to look for bad model results.

They are helpful to look for dynamically active regions so that you can avoid putting a nesting zone there. (that is: we want to put the nesting zone in areas where the flow fluctuates on timescales >> 1h to avoid aliasing and false shocks)

#### qc_gif
Plots any field stored on nodes, either as a sigma-layer movie, a z-level movie or a transect movie. It accepts a filename, a folder or a filelist as input.
```python
  >>> import fvtools.plot.qc_gif as qc

  >>> # sigma level movie
  >>> qc.main(folder='output01', var=['salinity', 'temp'], sigma=0)

  >>> # z-level movie
  >>> qc.main(filelist='filelist.txt', var=['salinity', 'temp'], z=10)

  >>> # transect movie
  >>> # --> either with a transect.txt input file with lon lat as colums
  >>> qc.main(fname='output01/casename.nc', var=['salinity', 'temp'], section='transect.txt')

  >>> # --> or as graphical input
  >>> qc.main(folder='output01', var=['salinity', 'temp'], section=True)

```

#### qc_gif_uv
As the name suggests, this creates quick georeferenced gifs of velocity fields. It accepts filelist and folder, as well as sigma and z.
```python
  >>> import fvtools.plot.qc_gif_uv as qc

  >>> # sigma level movie
  >>> qc.main(folder='output01', sigma=0)

  >>> # z level movie
  >>> qc.main(folder='output01', z=10)

```

## The mesh object - FVCOM_grid
The mesh object is the main interface for a quick look at an FVCOM grid. It reads a variety of input formats ("casename_xxxx.nc, casename_restart_xxxx.nc", "M.npy", "M.mat", "casename.2dm") and provide simple functions to look at a mesh, look at results from an experiment and return other useful forms of the mesh (i.e. .2dm file).

Example use:
```python
  >>> from fvtools.grid.fvcom_grd import FVCOM_grid
  >>> M = FVCOM_grid("casename_0001.nc")

  >>> # Get a quick summary of all attributes and functions:
  >>> print(M)

  >>> # See the mesh
  >>> M.plot_grid()

  >>> # Write the mesh to a .2dm file so that you can edit it in SMS
  >>> M.write_2dm()

```

You can georeference the mesh using geoplot
```python
  >>> from fvtools.plot.geoplot import geoplot
  >>> import matplotlib.pyplot as plt
  >>> gp = geoplot(M.x, M.y)
  >>> plt.imshow(gp.img, extent=gp.extent)
  >>> M.plot_grid()

```

### Plot results
Data stored on nodes can be plotted on visible node-based control volumes, for example:
```python
  >>> plt.imshow(gp, extent=gp.extent)
  >>> M.plot_cvs(M.h)

```
alternatively plot on triangle patches (much faster the first time):
```python
  >>> plt.imshow(gp, extent=gp.extent)
  >>> M.plot_field(M.h)

```

### Check mass conservation
```python
  >>> tracer = d['fabm_tracer'][:]
  >>> for i in range(len(tracer[:,0,0])):
  ...   mass = np.sum(M.node_volume * tracer[i,:].T)
  ...   plt.scatter(i, mass)

```

### Interpolate data to z-levels
```python
  >>> # Get salinity at 50 m depth and look at it
  >>> from netCDF4 import Dataset
  >>> data = Dataset('casename_0001.nc')
  >>> salt = data['salinity'][0,:]
  >>> salt_50m = M.interpolate_to_z(salt, 50)
  >>> M.plot_contour(salt_50m)

```


# Doctests

```bash
python -m doctest -v README.md
```
