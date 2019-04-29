import rasterio
import fiona
import rasterio.warp as warp
from rasterio.mask import mask
import os
from glob import glob
import numpy as np
from shapely.geometry import mapping, shape
from affine import Affine
from pyproj import Proj, transform
from s2cloudless import S2PixelCloudDetector



def createFootprint(pathname):
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
			return mask_only_raster, kwargs
		else:
			return landmasked_raster, kwargs


def apply_cloudmask(S2DotSafefp=None, bands=None, threshold=0.4, dilation_size=1,
					average_over=1, resolution_safe=100, save_mask=None, crs=None,
					affine=None):
	"""
	Function for applying cloud mask to either S2 product in safe format or a multiple rasters
	in numpy ndarray format. ndarray format is reccomended
	:param S2DotSafefp: path to S2 file
	:param bands: ndarray (N, H, W, B) N is the number of rasters, H is height, W is width, B are bands
	:param resolution_safe: Pixel resolution for cloud mask for .safe format in meters,
							recommended to use higher than 60 as it's much more efficient.
	:return: cloud mask (N, H, W) cloud mask for each raster, if .safe format the output shape will be
			(1, H, W)

	"""
	if S2DotSafefp is not None:
		res = np.array([60, 10, 10, 20, 10, 20, 60, 60, 20, 20])
		bands_ = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
		S2rasters = [glob(os.path.join(S2DotSafefp, 'GRANULE', 'L*', 'IMG_DATA','*{}*'.format(band)))[0]
					 for band in bands_]

		afftr = res/resolution_safe
		S2_cloud_bands = []

		for i, raster in enumerate(S2rasters):
			with rasterio.open(raster) as src:
				arr = src.read().astype('float32')/10000
				aff = src.transform
				new_aff = Affine(aff.a/afftr[i], aff.b, aff.c,
								 aff.d, aff.e/afftr[i], aff.f)

				newarr = np.empty(shape=(arr.shape[0],
										 int(round(arr.shape[1]*afftr[i])),
										 int(round(arr.shape[2]*afftr[i]))),dtype='float32')
				warp.reproject(arr, newarr,
							   src_transform=aff,
							   dst_transform=new_aff,
							   src_crs=src.crs,
							   dst_crs=src.crs)
				S2_cloud_bands.append(newarr)

		S2_cloud_bands = np.concatenate(S2_cloud_bands, axis=0)

		S2_cloud_bands = np.expand_dims(S2_cloud_bands, axis=0)

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
		raise ValueError("You need to provide a path to a Sentinel-2 .safe file, or a numpy ndarray consisting of the bands")







