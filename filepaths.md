# Betzy

## River positions
```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/RiverData/LargeRivers_030221.mat
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/RiverData/LargeRivers.mat
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/RiverData/SmallRivers.mat
```

## River runoff for each "vassdrag" (catchment area)
```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/riverdata_2013-2020.dat
```

## River temperatures
- All temperature-related data is stored under: /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
  - Replace .csv files with new ones when we get never data (remove the old one to avoid duplicates).
```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/RiverTemperature.mat
```
! NB: Special case, compiled for Finnmark, will only (?) work for BuildRivers.m

```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
```
- .csv: raw measured river temperatures. Format: "vassdragsområde.kystfelt_river_name.csv" (but be aware, NVE may have some typos for the "kystfelt"s)
- e.g.: /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/46.4_Bondhus.csv
- .npy: compiled river temperature files for vassdrag.
```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/PO7_temperatures.npy
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/PO10_temperatures.npy
```

## Coastline polygons
```
/cluster/shared/NS9067K/apn_backup/Kystlinje/Norgeskyst.txt
```

## Topography files (raw data from bathymetry.py, similarly named .npy files are also available)
```
/cluster/shared/NS9067K/apn_backup/Topo/FinnmarkTopo.txt
/cluster/shared/NS9067K/apn_backup/Topo/NNtopo.txt
/cluster/shared/NS9067K/apn_backup/Topo/NordNorgeTopo.txt
/cluster/shared/NS9067K/apn_backup/Topo/NT_topo.txt
/cluster/shared/NS9067K/apn_backup/Topo/Topo_Vestlandet.txt
/cluster/shared/NS9067K/apn_backup/Topo/TromsTopo.txt
/cluster/shared/NS9067K/apn_backup/Topo/VestlandetTopo.txt
```

## Tides
```
/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Tides/TPXO/olean/
- /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Tides/TPXO/olean/DATA/
- /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Tides/TPXO/olean/TPXO9_atlas/TPXO9_atlas_v5/
```

# Stokes

## Bathymetry data: For description see `fvtools/bathymetry/README`

```
/data/FVCOM/Setup_Files/bathymetry/gridded50_xyz
/data/FVCOM/Setup_Files/bathymetry/Norgeskyst.txt
/data/FVCOM/Setup_Files/bathymetry/norshelf_data
/data/FVCOM/Setup_Files/bathymetry/primaer_data_xyz
/data/FVCOM/Setup_Files/bathymetry/langfjorden_baty
```

## Topography
```
/data/Gridding/Topo/Finnmark_topo
/data/Gridding/Topo/Finnmark_topo.npy
/data/Gridding/Topo/FinnmarkTopo.npy
/data/Gridding/Topo/FinnmarkTopo.txt
/data/Gridding/Topo/More_bathy.npy
/data/Gridding/Topo/More_bathy.txt
/data/Gridding/Topo/NNtopo.npy
/data/Gridding/Topo/NNtopo.txt
/data/Gridding/Topo/NordNorgeTopo.npy
/data/Gridding/Topo/NordNorgeTopo.txt
/data/Gridding/Topo/nye_bunndata_oct2020
/data/Gridding/Topo/SorNorgeTopo.npy
/data/Gridding/Topo/SorNorgeTopo.txt
/data/Gridding/Topo/troms_topo.npy
/data/Gridding/Topo/trondelag_topo.npy
/data/Gridding/Topo/VestlandetTopo.npy
/data/Gridding/Topo/VestlandetTopo.txt
```

## Tides
All files under
```
/data/Tides/TPXO9_atlas/TPXO9_atlas_v5/
```

# fvtools
```
fvtools/rivers/Kystserier.csv
```
