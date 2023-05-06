# On the workflow in this repository

This repo has two main branches: `master` and `dev`. Master is protected (meaning you have to open a merge request to get stuff in there). `dev` is for active development and is occasionally merged into `master`.

# Setup
We recommend using a singularity container, a Singularity.def definitions file is provided. Create a singularity image from a linux terminal on a machine where you have sudo privileges, and type:
```
sudo singularity build python.sif Singularity.def
```
You will need Singularity.def and requirements.txt in the directory from where you build this image. Activate the image by calling;
```
singularity shell python.sif
```

Bind paths and mounts to the image, like this on sigma2 clusters
```
singularity shell --bind /cluster python.sif
```
or like this on Stokes
```
singularity shell --bind "/work,/data" /home/hes/python.sif
```

Once having activated the singularity shell, you're effectively running a Ubuntu machine with python and all python modules necessary to use fvtools interatively on the cluster. I like to work in ipyton, and this simply just need to call it to get started;
```singularity
ipython
```
Make sure to add your `fvtools` folder to your path before attempting to follow the examples listed herein.

# fvtools - tools to interact with FVCOM data
a variety of scripts to interact with FVCOM data before, during and after a model run.

[[_TOC_]]


# General idea
These scrips have been developed to emulate the workflow from fvcom_toolbox/fvtools in MATLAB

`We always`:
- Prepare our mesh in SMS, where we write a `.2dm` file with a nodestring on the outer boundaries
- Use high-resolution bathymetry datasets downloaded from Kartverket with approval from the Norwegian Navy.
- Use atmospheric forcing from the AROME configuration "MetCoOp" which is accessible on thredds.met.no

`We most of the time`:
- Work in Norway, where input data are most easilly accessible in `UTM33W` coordinates
- Nest into existing `FVCOM` experiments using files stored on `Betzy` or on `Stokes`

`We sometimes`:
- Nest into larger domain ROMS models operated by the met office
  - `NorKyst800` is an 800 m ROMS model configured for near-coast modelling.
  - `NorShelf2.5km` is an 2500 m resolution data assimilated ROMS model configured for shelf modelling.

All standard scripts can be called via the `main` function, which will create input/forcing files following a standardized setup:
- `fvtools.pre_pro.BuildCase.main` 
  - Quality controls the mesh (not complete, some rare cases such as nodes connected to two boundaries will not be detected yet)
  - Smooths raw bathymetry to desired `rx0` value (i.e. following MetOffice ROMS routines)
  - Returns estimate of necessary CFL criteria
  - Creates FVCOM.dat input files
    - `casename_cor.dat`
    - `casename_grd.dat`
    - `casename_obc.dat`
    - `casename_spg.dat`

- `fvtools.pre_pro.BuildRivers.main` creates river forcing for your experiment
- `fvtools.nesting.get_ngrd.main` cuts out a nestingzone mesh from the main mesh
  - for fvcom to fvcom nested domains, this routine will dump the `casename_bathymetry.dat` file to your `input` folder 
- `fvtools.atm.read_metCoop.main` interpolates atmospheric forcing to your domain

Nesting is for ROMS nested and FVCOM nested models are done using:
- `fvtools.nesting.roms_nesting_fg.main`
  - will dump a `casename_bathymetry.dat` file to your `input` folder
- `fvtools.nesting.fvcom2fvcom_nesting.main`

# Workflow:
A typical Akvaplan FVCOM experiment goes through some standard steps:

## Preparing an experiment
1. Create a folder called `casename` and a subfolder of it called `input`.
2. Put a `casename_sigma.dat` file into `casename/input`, for example a TANH sigma coordinate:
```dat
NUMBER OF SIGMA LEVELS = 35
SIGMA COORDINATE TYPE = TANH
DU = 2.5
DL = 0.5
```

### Preparing the mesh
#### Writing .dat grid files
You have a `casename.2dm` file (either from smeshing or from SMS), create the FVCOM grid input files using `BuildCase`:
```python
import fvtools.pre_pro.BuildCase as bc
bc.main('cases/inlet/casename.2dm', 'bathymetry.txt')
```
`Mesh quality`: BuildCase prepares the grid before a simulation. It checks if there are invalid triangles (triangles with more than one side facing the boundary), and will remove such triangles if present.

`Bathymetry`: BuildCase smooths the bathymetry to a  [rx0](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwil_uzfvL39AhWPnYsKHZREAtAQFnoECAUQAQ&url=http%3A%2F%2Fmathieudutour.altervista.org%2FPresentations%2FSteepnessPresExt.pdf&usg=AOvVaw1bBZEBsmkvWmSeCkxlrE3y) factor less than `rx0max`. 

`Boundary setup`: It finds the `OBC`nodes, the `sponge nodes` and sets a `sponge factor`if asked to (defaults to 0).

`BuildCase` returns `casename_*.dat` FVCOM input files to the `input` folder, and a file called `M.npy` used by other setup routines.

#### The mesh in the nesting zone
fvtools support two nesting types:
- From `ROMS` (either `NorKyst` or `NorShelf`)
- From `FVCOM` (from any `FVCOM` model that overlaps with this mesh)

```python
import fvtools.nesting.get_ngrd as gn

# ROMS-FVCOM nesting:
# R is the nestingzone width (R measured in meters). This is typically approximately 4.5 times the mesh resolution at the OBC.
gn.main('M.npy', R=4.5*800)

# FVCOM-FVCOM nesting:
gn.main('M.npy', mother='mother_fvcom.nc')

# These routines will create a ngrd.npy file in your working directory
```

### Nesting (OBC forcing)
Interpolating ocean state to the FVCOM nesting zone.

`FVCOM to FVCOM nesting` requires a [filelist](https://source.coderefinery.org/apn/fvtools/-/blob/hes/README.md#a-filelist-linking-to-fvcom-results).
```python
import fvtools.nesting.fvcom2fvcom_nesting as f2f
f2f.main(ngrd = 'ngrd.npy', 
         fileList = 'fileList.txt', 
         output_file = '/input/my_experiment_nest.nc', 
         start_time = '2018-01-01-00', 
         stop_time = '2018-02-01-00')

```

`ROMS-FVCOM nesting` will automatically find the forcing it need at thredds.met.no:
```python
import fvtools.nesting.roms_nesting_fg as rn
rn.main('M.npy', 'ngrd.npy', './input/casename_nest.nc', '2018-01-01-00', '2018-02-01-00', mother='NS')
```

MET Norway does _not_ guarantee data continuous in time. We need to keep the tides continuous, for that we use `fvtools.nesting.fill_nesting.gaps`. It performs a tidal analysis for `zeta` at every `node`, `ua, va` at every `cell`, and `u, v` in every `sigma layer` at every `cell` to fill gaps. Temperature and salinity are linearly interpolated. See examples/run_gap_filler.py and run_python.sh for example use.

`ROMS2FVCOM currently supports these ROMS models/setups`
- Met office NorKyst `MET-NK`
- IMR NorKyst `HI-NK`
- Hourly NorShelf `H-NS`
- Daily averaged NorShelf `D-NS`

Other ROMS experiments can be supported by adding a ROMS reader to `fvtools.grid.roms_grd`.

### River runoff
River runoff is given for geographical catchment areas ids (vassdrag), the ids are mapped at [nve atlas](https://atlas.nve.no/)
- river temperatures are read from NVE text-files. 
- a new "FVCOM-mother" grid requires you to compile a new rivertemp.npy file.

Temperature files used to force the FVCOM-mother models are stored on the Stokes and Betzy, use these when nesting in a smaller model.
  - Stokes: `/data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/`
  - Betzy:  `/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/`

```python
import fvtools.pre_pro.BuildRivers as br
br.main('2018-01-01-00', '2018-02-01-00', vassdrag, temp='compile')
br.main('2018-01-01-00', '2018-02-01-00', vassdrag, temp='fvcom_mother_temperatures.npy')

```
The routine writes a file called `RiverNamelist.nml` and `riverdata.nc` to your working directory. Put these in the `input` folder.

### Atmospheric forcing
We use the MetCoOp-AROME model for atmospheric forcing.
```python
import fvtools.atm.read_metCoop as rm
rm.main("M.npy", "./input/casename_atm.nc", "2018-01-01-00", "2018-02-01-00")
```

### Initial conditions
#### interpol_restart
Interpolates inital fields from a FVCOM mother to a restart file.
- using an existing `restartfile.nc`
- make a restartfile for a `startdate`

```python
import fvtools.pre_pro.interpol_restart as ir
ir.main(childfn="casename_restart_0001.nc", 
        filelist="filelist_restart.txt")

# alternatively if you need to make a restart file
ir.main(startdate="2023-01-31-00", 
        filelist="filelist_restart.txt")

# If you don't have a filelist
ir.main(childfn="casename_restart_0001.nc", 
        result_folder="./mother_folder/output01/", 
        name = "mother_restart")
```

#### interpol_roms_restart
Interpolated initial fields from a ROMS model to your restartfile, but be warned: the model may crash if you set `uv=True`.
```python
import fvtools.pre_pro.interpol_roms_restart as ir 
ir.main("casename_restart_0001.nc", "NS", uv=True)
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
python fvcom_make_filelist.py -d folders.txt -s PO10 -f fileList.txt
```

### Take a quick look at the results
`qc_gif` and `qc_gif_uv` creates animations of scalar or velocity fields between `[start, stop]` (or the entire timespan if not specified).

They are helpful to look for dynamically active regions so that you can avoid putting a nesting zone there. (that is: we want to put the nesting zone in areas where the flow fluctuates on timescales >> 1h to avoid aliasing and false shocks)

#### qc_gif
Plots any field stored on nodes, either as a sigma-layer movie, a z-level movie or a transect movie. It accepts a filename, a folder or a filelist as input.
```python
import fvtools.plot.qc_gif as qc
qc.main(folder='output01', var=['salinity', 'temp'], sigma=0)
qc.main(filelist='filelist.txt', var=['salinity', 'temp'], z=10)

# transect movie (from file by setting setting=file.txt, or graphically by setting section = True)
qc.main(fname='output01/casename.nc', var=['salinity', 'temp', 'tracer_01'], section='transect.txt')
qc.main(folder='output01', var=['salinity', 'temp'], section=True)
```

#### qc_gif_uv
Creates quick georeferenced gifs of velocity fields. It accepts `filelist` and `folder`, as well as `sigma` and `z`.
```python
import fvtools.plot.qc_gif_uv as qc
qc.main(folder='output01', sigma=0)
qc.main(folder='output01', z=10)

```
## The mesh object - FVCOM_grid
The mesh object is the main interface for a quick look at an FVCOM grid. It reads a variety of input formats:
- `casename_xxxx.nc`
- `casename_restart_xxxx.nc`
- `M.npy`
- `M.mat`
- `casename.2dm`
- `smeshing.txt`

Provides simple functions to look at a mesh and results. Methods to export the mesh, write `*.dat` input files etc.

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
Data stored on nodes can be plotted on node-based control volumes patches, for example:
```python
M.georeference()
M.plot_cvs(M.h)

```
alternatively plot on triangle patches (much faster the first time):
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
from netCDF4 import Dataset
salt = M.load_netCDF('casename_0001.nc', 'salinity' 0)
salt_50m = M.interpolate_to_z(salt, 50)
M.georeference()
M.plot_contour(salt_50m)
```
