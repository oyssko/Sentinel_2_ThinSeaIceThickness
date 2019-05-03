import rasterio
import os
from glob import glob
from rasterio.warp import Resampling
import rasterio.warp as warp


def Acolite_reproject_stack_bands(srcPath, s2path, destBands, destination):
    """
    Reproject the ACOLITE output to fit to a overlapping raster

    :param srcPath: the overlapping raster
    :param s2path: path to the ACOLITE output
    :param destBands: destination for the reprojected bands
    :param destination: destination for the final stacked output raster
    :return: None
    """
    if os.path.basename(glob(os.path.join(s2path,'S2*'))[0])[:3]=='S2B':
        bands = ['442', '492', '559', '665', '704', '739', '780', '833', '864', '1610', '2186', ]
    else:
        bands = ['443', '492', '560', '665', '704', '740', '783', '833', '865', '1614', '2202']

    band_list = []
    if not os.path.exists(destBands):
        os.mkdir(destBands)
    with rasterio.open(srcPath) as src:
        kwargs = src.profile
        for band in bands:
            rhos_ = glob(os.path.join(s2path, '*rhos_' + band + '.tif'))[0]
            base_rhos_ = os.path.basename(rhos_)[0:-4]
            dest_rhos_ = os.path.join(destBands, base_rhos_ + 'reproject.tif')
            band_list.append(dest_rhos_)
            print("reprojecting: {}".format(base_rhos_))
            with rasterio.open(rhos_) as s2:
                kwargs.update({'nodata': s2.profile['nodata'],
                               'driver': 'GTiff',
                               'dtype': s2.profile['dtype']
                               })
                with rasterio.open(dest_rhos_, 'w', **kwargs) as s2dst:
                    warp.reproject(
                        rasterio.band(s2, 1),
                        rasterio.band(s2dst, 1),
                        resampling=Resampling.average
                    )
        print("Stacking layers to: {}".format(destination))
        with rasterio.open(band_list[1]) as src0:
            kwargs = src0.profile
            kwargs.update({'count': len(band_list),
                           'nodata': src0.profile['nodata']})
        with rasterio.open(destination, 'w', **kwargs) as dst:
            for id, layer in enumerate(band_list, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))

srcPath = r"C:\Users\oyste\OneDrive\Shared\UiT skole\MastersFolder\raster_code\sentinel4thinice_navgem_500\navgem\500\20190311_103537_103837_slstr_tti-color_500.tif"
path = r"E:\MastersProjectData\SIT_S2\20190311_103537_103837_slstr_tti-color_500"
s2path = os.path.join(path, 'merged_ACOLITE')
destination = os.path.join(s2path, 'merged.tif')
destBands = os.path.join(s2path, 'reproject_bands')

Acolite_reproject_stack_bands(srcPath, s2path, destBands, destination)
