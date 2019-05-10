import rasterio
from raster_utils import read_S2TOA, read_acolite, apply_landmask, apply_acolite, apply_cloudmask
import numpy as np
import os
from S2_SIT import S2_SIT_model #--> function still need to be made, more data, and get
                                # auxillary data and interpolate with S2 data
import argparse

#from raster_utils import apply_landmask, apply_cloudmask

# 1. Cloudmask - requires TOA reflectances, save and apply to BOA (resample to Acolite output, res)
# 2. DOS/atmospheric correction - Acolite (10, 20, or 60 meter resolution, choose one closes to desired
#    resolution)
# 3. Final resampling to desired resolution
# 4. landmask
# 5. water mask TODO: Find suitable water mask using NDWI method, for TOA reflectances
# 6. The pixels left should be sea ice, apply thin SIT model
# 7. Add mask values for land, open water, clouds and SIT values and save raster

parser = argparse.ArgumentParser(description=__doc__)
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")

required.add_argument('--S2_safe_fp', help="Path to Sentinel 2 .safe file", required=True)
required.add_argument('--output_folder', help="output folder for SIT estimate raster", required=True)
required.add_argument('--pixel_res', type=int, help="Pixel resolotion of output, either 10, 20 or 60 m. Default=20",
                      default=60)
required.add_argument('--BOA', help="Use BOA reflectance values from Acolite", action='store_true')
required.add_argument('--display', help='Display results', action='store_true')

args = parser.parse_args()
raster_fp = args.S2_safe_fp
BOA = args.BOA
display = args.display
outputdest = args.output_folder
pixel_res = args.pixel_res

tempfolder = r"temp"
masks = []
if not os.path.exists(tempfolder):
    os.mkdir(tempfolder)
print("Reading S2 raster:")
S2_TOAbands, temp_raster= read_S2TOA(raster_fp, pixel_res=pixel_res,temp=tempfolder, resample='average')
"""Cloudmask"""
print("Extracting cloud mask:")
valid_cloud_mask_bands = [0,1,3,4,7,8,9,10,11,12]
cloud_bands = np.expand_dims(S2_TOAbands[..., valid_cloud_mask_bands], axis=0)
cloud_mask = apply_cloudmask(S2DotSafefp=None,bands=cloud_bands)
cloud_mask = cloud_mask[0]==1

print("Extracting cloud mask done.")
"""Landmask"""
print("Extracting land mask:")
land_mask = apply_landmask(temp_raster, detail=True, mask_only=True)
if land_mask is None:
    land_mask = np.zeros_like(S2_TOAbands[..., 0]) == 1
print("Extracting land mask done")
"""If you want to estimate on boa values"""
if BOA:
    del S2_TOAbands
    print("Apply ACOLITE atmospheric correction")
    acolite_out_rel = os.path.join(tempfolder, 'ACOLITE')
    acolite_out = os.path.abspath(acolite_out_rel)
    apply_acolite(raster_fp, acolite_out, S2_target_res=pixel_res)
    print("Read BOA reflectances")
    S2_BOA_bands = read_acolite(acolite_out)

    for the_file in os.listdir(acolite_out):
        file_path = os.path.join(acolite_out, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    os.rmdir(acolite_out)
    print("Estimate thin SIT from model")
    SIT_estimate = S2_SIT_model(S2_BOA_bands,
                                cloud_mask=cloud_mask,
                                land_mask=land_mask,
                                BOA=True, display=display)

else:
    """Else, use TOA values"""
    print("Estimate thin SIT from model")
    SIT_estimate = S2_SIT_model(S2_TOAbands,
                                cloud_mask=cloud_mask,
                                land_mask=land_mask,
                                BOA=False, display=display)
safe_name = os.path.basename(raster_fp)
safe_conf = safe_name.split('_')
datetime = safe_conf[2].split('T')
time = datetime[0]+'_'+datetime[1]
if BOA:
    SIT_raster = time+'_'+safe_conf[5]+'_'+'SIT_estimate_MSI_BOA'+str(pixel_res)+'m'+'.tif'
else:
    SIT_raster = time + '_' + safe_conf[5] + '_' + 'SIT_estimate_MSI_TOA' + str(pixel_res) + 'm' + '.tif'

output_raster = os.path.join(outputdest, SIT_raster)
print("Writing SIT estimate to ", outputdest)
with rasterio.open(temp_raster) as src:
    kwargs = src.profile
    kwargs.update({"driver": 'GTiff',
                   "dtype": SIT_estimate.dtype})
    with rasterio.open(output_raster, 'w', **kwargs) as dst:
        dst.write(SIT_estimate, 1)

os.remove(temp_raster)
os.rmdir(tempfolder)







