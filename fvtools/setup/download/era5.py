"""Download ERA5 forcing via CDS API."""
import numpy as np
from datetime import datetime
from fvtools.grid.fvcom_grd import FVCOM_grid

def main(model: FVCOM_grid, start: datetime, stop: datetime) -> None:
    '''
    Download ERA5 data using the cdsapi

    model: pyfvcom.preproc.Model instance
    '''
    import cdsapi
    # Define what data you will download
    years  = np.array([start.year, stop.year])
    months = np.array([start.month, stop.month])
    if np.diff(years) == 0:
        years = [f'{years[0]}']
        months = [f'{m:2d}' for m in range(months[0], months[1]+1)]
    else:
        years = [f'{y}' for y in range(years[0], years[1]+1)]
        months = [f'{m:2d}' for m in range(1,13)]

    variables = [
        #"10m_u_component_of_wind",
        #"10m_v_component_of_wind",
        #"2m_temperature",
        #"surface_pressure",
        #"total_precipitation",
        "mean_surface_downward_long_wave_radiation_flux",
        "mean_surface_net_short_wave_radiation_flux",
        "total_cloud_cover",
        "evaporation",
        "2m_dewpoint_temperature",
        "land_sea_mask"
    ]

    dataset = "reanalysis-era5-single-levels"
    for variable in variables:
        print(f'Downloading {variable}')
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable],
            "year": years,
            "month": months,
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [model.lat.max() + 0.5, model.lon.min() - 0.5, model.lat.min() - 0.5, model.lon.max() + 0.5]
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download()
