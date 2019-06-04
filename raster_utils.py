import rasterio
import fiona
import rasterio.warp as warp
from rasterio.mask import mask
from rasterio.warp import Resampling
import os
from glob import glob
import numpy as np
from shapely.geometry import mapping, shape
from affine import Affine
from pyproj import Proj, transform
from s2cloudless import S2PixelCloudDetector
import subprocess


def resampling_method(method):
    if method == 'nearest':
        return Resampling.nearest
    elif method == 'bilinear':
        return Resampling.bilinear
    elif method == 'cubic':
        return Resampling.cubic
    elif method == 'cubic_spline':
        return Resampling.cubic_spline
    elif method == 'lanczos':
        return Resampling.lanczos
    elif method == 'average':
        return Resampling.average
    elif method == 'mode':
        return Resampling.mode
    elif method == 'max':
        return Resampling.max
    elif method == 'min':
        return Resampling.min
    elif method == 'med':
        return Resampling.med
    elif method == 'q1':
        return Resampling.q1
    elif method == 'q3':
        return Resampling.q3
    else:
        raise ValueError("Invalid method, does not exist, check rasterio.warp.Resampling for reference.")

def read_S2TOA(S2_SAFE_fp, pixel_res, temp=None, resample='average'):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09','B10', 'B11', 'B12']
    res = np.array([60, 10, 10, 10, 20, 20, 20, 10, 20, 60,60, 20, 20])
    # list of rasters
    S2rasters = [glob(os.path.join(S2_SAFE_fp, 'GRANULE', 'L*', 'IMG_DATA', '*{}*'.format(band)))[0]
                 for band in bands]
    afftr = res / pixel_res
    S2_bands = []
    for i, band in enumerate(S2rasters):
        print("Band ", i + 1)
        with rasterio.open(band) as src:
            arr = src.read().astype('float32') / 10000
            kwargs = src.profile
            if res[i] == pixel_res:
                S2_bands.append(arr)
                if temp is not None:
                    temp_raster = os.path.join(temp, 'temp_raster.tif')
                    kwargs.update({'width': arr.shape[1],
                                   'height': arr.shape[2],
                                   "driver": 'GTiff',
                                   "dtype": arr.dtype,
                                   "nodata": src.profile['nodata']})
                    with rasterio.open(temp_raster, 'w', **kwargs) as dst:
                        dst.write(arr)
                continue
            else:
                aff = src.transform
                new_aff = Affine(aff.a / afftr[i], aff.b, aff.c,
                                 aff.d, aff.e / afftr[i], aff.f)

                kwargs.update({"transform": new_aff,
                               "driver": 'GTiff',
                               "dtype": arr.dtype,
                               "nodata": src.profile['nodata']})
                newarr = np.empty(shape=(arr.shape[0],
                                         int(round(arr.shape[1] * afftr[i])),
                                         int(round(arr.shape[2] * afftr[i]))), dtype='float32')
                warp.reproject(arr, newarr,
                               src_transform=aff,
                               dst_transform=new_aff,
                               src_crs=src.crs,
                               dst_crs=src.crs,
                               resampling=resampling_method(resample))
                S2_bands.append(newarr)




    S2_bands = np.concatenate(S2_bands, axis=0).transpose(1, 2, 0)
    if temp is not None:
        return S2_bands, temp_raster
    else:
        return S2_bands

def apply_acolite(S2_SAFE_fp,output, S2_target_res):

    """
    Create settings file for ACOLITE AC processor for multiple tiles and saves it in the
    working directory as "aculite_settings.txt". For any other settings, look at the aculite_config.txt
    in aculite_py_win/config folder

    :param inputfolder: folder with all the S2 level 1 products in SAFE format
    :param output: Output folder for
    :param s2_target_res: Resolotion for processed produce
    :param merge_tiles: if you do not wish to merge, make this False
    :return: settings
    """
    inputfiles = 'inputfile='+S2_SAFE_fp+'\n'
    outputfile = 'output=' + output + '\n'
    l2w = 'l2w_parameters=rhos_*\n'
    target_res = 's2_target_res=' + str(S2_target_res) + '\n'
    settings = inputfiles + outputfile + l2w + target_res

    with open("acolite_settings.txt", 'w') as dst:
        dst.write(settings)

    cwd = os.getcwd()
    acolitefp = r'acolite_py_win/dist/acolite'
    os.chdir(acolitefp)
    comm = r'acolite.exe --cli --settings="..\..\..\acolite_settings.txt"'
    subprocess.run(comm, shell=True)
    os.chdir(cwd)

def read_acolite(Acolite_fp):

    if os.path.basename(glob(os.path.join(Acolite_fp,'S2*'))[0])[:3]=='S2B':
        bands = ['442', '492', '559', '665', '704', '739', '780', '833', '864', '1610', '2186', ]
    else:
        bands = ['443', '492', '560', '665', '704', '740', '783', '833', '865', '1614', '2202']
    S2_BOA_bands = []
    """
    afftr = 10 / pixel_res
    for i, band in enumerate(bands):
        print("Band ", i+1, ':', bands[i])
        rhos_ = glob(os.path.join(Acolite_fp, '*rhos_' + band + '.tif'))[0]
        with rasterio.open(rhos_) as acsrc:
            arr = acsrc.read()
            aff = acsrc.transform
            new_aff = Affine(aff.a / afftr, aff.b, aff.c,
                             aff.d, aff.e / afftr, aff.f)
            kwargs = acsrc.profile
            kwargs.update({"transform": new_aff,
                           "driver": 'GTiff',
                           "dtype": arr.dtype,
                           "nodata": acsrc.profile['nodata']})
            newarr = np.empty(shape=(arr.shape[0],
                                     int(round(arr.shape[1] * afftr)),
                                     int(round(arr.shape[2] * afftr))), dtype='float32')
            warp.reproject(arr, newarr,
                           src_transform=aff,
                           dst_transform=new_aff,
                           src_crs=acsrc.crs,
                           dst_crs=acsrc.crs,
                           resampling=resampling_method(resample))
            S2_BOA_bands.append(newarr)
            print(newarr.shape)
            #S2_BOA_bands.append(arr)
    """
    for i, band in enumerate(bands):
        print("Band ", i + 1, ':', bands[i])
        rhos_ = glob(os.path.join(Acolite_fp, '*rhos_' + band + '.tif'))[0]
        with rasterio.open(rhos_) as acsrc:
            arr = acsrc.read()
            S2_BOA_bands.append(arr)

    S2_BOA_bands = np.concatenate(S2_BOA_bands, axis=0).transpose(1, 2, 0)

    return S2_BOA_bands

def createFootprint(pathname):
    """
    Function for creating footprint from geotiff raster file

    :param pathname: path to geotiff raster
    :return: geojson like object defining the footprint
    """
    # Read raster
    with rasterio.open(pathname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read(1)  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong', datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)

    lonlat = np.zeros((longs.shape + (2,)))

    lonlat[..., 0] = longs
    lonlat[..., 1] = lats

    # Take only the edge pixels of the raster
    right = lonlat[:, 0, :][0::100]  # right
    down = lonlat[-1, :, :][0::100]  # down
    left = np.flip(lonlat[:, -1, :][0::100], axis=0)  # left
    up = np.flip(lonlat[0, :, :][0::100], axis=0)  # up
    up[-1] = right[0]
    footprint_arr = np.concatenate((right, down, left, up), axis=0).tolist()
    # Create geojson like file
    geom = {
        "type": "Polygon",
        "coordinates": [

        ]
    }
    # Append the footprint coordinates to the json
    geom["coordinates"].append(footprint_arr)

    return geom


def apply_landmask(rasterfp, detail=True, keep_land=False, mask_only=False):
    """
    Function for applying landmask, using a landmask shapefile in the included folder

    rasterfp - path to raster
    detail - Get detailed high resolution land mask if True (slow, high memory), else using a more rough, but much faster
    and efficient
    keep_land - Mask out land (False), or keep only the land pixels (True)
    mask_only - return just the mask (True), or remove the masked part from the raster (False)
    """
    # create Footprint for raster
    footprint = createFootprint(rasterfp)

    if detail:
        landpoly = r"include/land_polygons.shp"
    else:
        landpoly = r"include/GSHHS_i_L1.shp"

    with fiona.open(landpoly) as shapefile:
        # Use shapely for finding intersections between raster and land shapefile
        # by checking the shape intersects with the created footprint
        # Saves memory by not loading the whole shape, though it takes some time
        intersections = [mapping(shape(feature['geometry'])) for feature in shapefile
                         if shape(feature['geometry']).intersects(shape(footprint))]
        # Get shapefiles crs
        crsShape = shapefile.crs

    # Check if there are any intersections, if there are none, just return a message.
    if len(intersections) == 0:
        print("There are no lands within the raster. Returning None")
        return None

    # If there are, apply the mask:
    else:
        # Use rasterio.warp's transform_geom to transform the intersection shapes crs to the rasters crs
        # then mask out the pixels that's within the shapes using rasterio.mask's mask method
        with rasterio.open(rasterfp) as src:
            kwargs = src.profile
            intersect_trans = [warp.transform_geom(crsShape, src.crs, land) for land in intersections]
            landmasked_raster, transform = mask(src, intersect_trans, invert=~keep_land)

        if mask_only:
            mask_only_raster = np.zeros_like(landmasked_raster)
            mask_only_raster[0] = (landmasked_raster[0] == 0).astype('uint16')
            return mask_only_raster
        else:
            return landmasked_raster


def apply_cloudmask(S2DotSafefp=None, bands=None, threshold=0.4, dilation_size=3,
                    average_over=1, resolution_safe=100, save_mask=None, crs=None,
                    affine=None):
    """
    Function for applying cloud mask to either S2 product in safe format or a multiple rasters
    in numpy ndarray format. ndarray format is reccomended

    :param S2DotSafefp: path to S2 file
    :param bands: ndarray (N, H, W, B) N is the number of rasters, H is height, W is width, B are bands
    :param resolution_safe: Pixel resolution for cloud mask for .safe format in meters,
                            recommended to use higher than 60 as it's much more efficient.
    :param dilation_size: dilation for cloud pixels
    :param threshold: threshold for the cloud probability
    :param average_over: neighborhood of averaging
    :param save_mask:
    :param crs: If save_mask => crs for the saved mask
    :param affine: Affine transformation for the saved mask
    :return: cloud mask (N, H, W) cloud mask for each raster, if .safe format the output shape will be
            (1, H, W)

    """
    # If the input is in safe format
    if S2DotSafefp is not None:
        # resolutions for each band
        res = np.array([60, 10, 10, 20, 10, 20, 60, 60, 20, 20])
        # Valid bands
        bands_ = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        # list of rasters
        S2rasters = [glob(os.path.join(S2DotSafefp, 'GRANULE', 'L*', 'IMG_DATA', '*{}*'.format(band)))[0]
                     for band in bands_]
        # Affine transformation list
        afftr = res / resolution_safe
        S2_cloud_bands = []

        # iterate through band rasters
        for i, raster in enumerate(S2rasters):
            with rasterio.open(raster) as src:
                # Divide by 10000 to get TOA values
                arr = src.read().astype('float32') / 10000
                aff = src.transform
                # Define new affine transform
                new_aff = Affine(aff.a / afftr[i], aff.b, aff.c,
                                 aff.d, aff.e / afftr[i], aff.f)
                # Empty array for destination
                newarr = np.empty(shape=(arr.shape[0],
                                         int(round(arr.shape[1] * afftr[i])),
                                         int(round(arr.shape[2] * afftr[i]))), dtype='float32')
                # Apply downsampling
                warp.reproject(arr, newarr,
                               src_transform=aff,
                               dst_transform=new_aff,
                               src_crs=src.crs,
                               dst_crs=src.crs)
                # Append to list
                S2_cloud_bands.append(newarr)
        # Concatenate the list and add a dim for compatibility with S2Cloudless function
        S2_cloud_bands = np.concatenate(S2_cloud_bands, axis=0)

        S2_cloud_bands = np.expand_dims(S2_cloud_bands, axis=0)
        # Apply cloudmask
        cloudmask = S2PixelCloudDetector(threshold=threshold,
                                         dilation_size=dilation_size,
                                         average_over=average_over,
                                         all_bands=False).get_cloud_masks(S2_cloud_bands)

        # save mask if path is specified
        if save_mask is not None:
            with rasterio.open(S2rasters[0]) as src:
                kwargs = src.profile
                kwargs['width'] = cloudmask.shape[1]
                kwargs['height'] = cloudmask.shape[2]
                kwargs['transform'] = new_aff
                kwargs['driver'] = 'Gtiff'
                kwargs['dtype'] = cloudmask.dtype
                kwargs['blockxsize'] = 128
                kwargs['blockysize'] = 128

                with rasterio.open(save_mask, 'w', **kwargs) as dst:
                    dst.write(cloudmask)
        # return the cloudmask as np ndarray
        return cloudmask
    # If valid bands are given
    elif bands is not None:

        cloudmask = S2PixelCloudDetector(threshold=threshold,
                                         dilation_size=dilation_size,
                                         average_over=average_over,
                                         all_bands=False).get_cloud_masks(bands)

        # save mask if path is specified
        if save_mask is not None and crs is not None:
            kwargs = {
                'driver': 'GTiff',
                'interleave': 'band',
                'tiled': True,
                'blockxsize': 128,
                'blockysize': 128,
                'nodata': 0,
                'dtype': cloudmask.dtype,
                'height': cloudmask.shape[1],
                'width': cloudmask.shape[2],
                'crs': crs,
                'transform': affine
            }

            with rasterio.open(save_mask, 'w', **kwargs) as dst:
                dst.write(cloudmask)
        elif save_mask is not None and crs is None:
            raise ValueError("To save the mask, crs needs to be provided")
        # return the cloudmask as np ndarray
        return cloudmask
    else:
        raise ValueError(
            "You need to provide a path to a Sentinel-2 .safe file, or a numpy ndarray consisting of the bands")

def Water_mask(S2_ndarray, threshold):

    pass

def apply_dos_all(S2_array):
    S2_dos = np.zeros_like(S2_array)
    for i in range(S2_dos.shape[-1]):
        S2_dos[..., i] = apply_dos(S2_array[..., i])

    return S2_dos

def apply_dos(S2_band):
    do = S2_band[S2_band!=0].min()
    S2_band[S2_band!=0] = S2_band[S2_band!=0]-do
    return S2_band

def estimate_albedo(S2_arr, BOA=False):
    weights = np.loadtxt('models/albedo_weights.txt')
    if BOA:
        weights_BOA = weights[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]
        S2_tot_alb = np.zeros((S2_arr.shape[0],S2_arr.shape[1]))
        for i in range(S2_arr.shape[2]):
            S2_tot_alb += S2_arr[..., i] * weights_BOA[i]
    else:
        S2_tot_alb = np.zeros((S2_arr.shape[0],S2_arr.shape[1]))
        for i in range(S2_arr.shape[2]):
            S2_tot_alb += S2_arr[..., i] * weights[i]

    return S2_tot_alb