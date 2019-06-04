# Sentinel_2_ThinSeaIceThickness
Ipython command line function for estimating thin sea ice thickness using a regression model for Sentinel 2.
Created for Øystein F. Skogvold's Master Thesis project, "Arctic Thin Sea Ice Thickness Models for Sentinel-2".

## Download

1. Clone or download repository
2. Create a virtual environment and activate
..* `>conda create -n venv python=3.6`
..* `>conda activate venv´
3. For installing the required packaged for running the scripts, first install the Rasterio package with the [guide](https://rasterio.readthedocs.io/en/stable/installation.html) from Rasterio documentation.
4. Install required python packages from requirements.txt
..* `pip install -r requirements.txt´
5. For atmospheric correction:
..1. Download the [ACOLITE atmospheric correction processor](https://odnature.naturalsciences.be/remsem/software-and-data/acolite) for windows
..2. Unzip the file and add the `acolite_py_win` folder to the repository folder
6. For detailed landmasking:
..1. Download the land polygons from this [link](https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip)
..2. Unzip the file and add `land_polygons.shp` to the `include` folder in the repository folder

You're set to run the scripts

## Usage
Example usage in ipython:

```
 run S2_Thin_SIT.py --S2_safe_fp "PATH TO S-2 .SAFE FOLDER" --output_folder "PATH TO OUTPUT FOLDER" --p
   ...: ixel_res 60 --BOA --tot_albedo --display
```
Usage is defined below:

```
usage: S2_Thin_SIT.py [-h] --S2_safe_fp S2_SAFE_FP --output_folder
                      OUTPUT_FOLDER [--pixel_res PIXEL_RES] [--BOA] [--DOS]
                      [--tot_albedo] [--display]

Module created for script run in IPython

required arguments:
  --S2_safe_fp S2_SAFE_FP
                        Path to Sentinel 2 .safe file
  --output_folder OUTPUT_FOLDER
                        output folder for SIT estimate raster
  --pixel_res PIXEL_RES
                        Pixel resolotion of output, either 10, 20 or 60 m.
                        Default=60
  --BOA                 Use BOA reflectance values from Acolite
  --DOS                 Apply DOS
  --tot_albedo          Estimate SIT from total albedo
  --display             Display results
```

The output is stored in 16 bit unsigned integers, also called DN values. To retrieve thin SIT values from 0 to 30, convert the DN values to float and divide by 1000.
After this, the cloud pixels have value of 40 and land pixels have value of 50.

### `config.json`
The configuration json file includes some hyperparameters for the cloud/land masking and resampling method:
```
{
  "cloudmask": {
    "average_over": 1,
    "dilation": 3,
    "threshold": 0.3
  },
  "landmask": {
    "detailed": true
  },
  "resampling": "average"
}
```
These are the default values. The resampling parameter must follow the methods from rasterio.warp.Resampling:
```Available warp resampling algorithms.
    The first 8, 'nearest', 'bilinear', 'cubic', 'cubic_spline',
    'lanczos', 'average', 'mode', and 'gauss', are available for making
    dataset overviews.
    'max', 'min', 'med', 'q1', 'q3' are only supported in GDAL >= 2.0.0.
    'nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos',
    'average', 'mode' are always available (GDAL >= 1.10).
    Note: 'gauss' is not available to the functions in rio.warp.
```




