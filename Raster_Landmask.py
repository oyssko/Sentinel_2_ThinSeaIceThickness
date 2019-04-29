import rasterio
import numpy as np
import os
import argparse
from raster_utils import apply_landmask

parser = argparse.ArgumentParser(description=__doc__)
optional = parser._action_groups.pop()
required = parser.add_argument_group("Required arguments")

required.add_argument("--src", help="Path to raster", required=True)
required.add_argument("--dst", help="Destination folder path", required=True)
required.add_argument("--detailed", help="Use detailed landmask",
					  action='store_true')
required.add_argument("--keep_land", help="Mask out the land",
					  action='store_true')
required.add_argument("--mask_only", help="Keep the mask only",
					  action='store_true')

args = parser.parse_args()
rasterfp = args.src
destination = args.dst
detail = args.detailed
keep_land = args.keep_land
mask_only = args.mask_only
	
landmask, kwargs = apply_landmask(rasterfp, detail=detail, keep_land=keep_land, mask_only=mask_only)
kwargs['driver']='GTiff'
basename, _ = os.path.basename(rasterfp).split('.')
new_name = basename + '_landmasked.'+'tif'
dst = os.path.join(destination, new_name)

with rasterio.open(dst , 'w', **kwargs) as dest:
	dest.write(landmask)


