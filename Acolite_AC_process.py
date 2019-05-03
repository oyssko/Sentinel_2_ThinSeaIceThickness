from glob import glob
import os
from lxml.etree import parse
from shapely.geometry import Polygon
import numpy as np
import subprocess

def _polygon_from_coords(coords, fix_geom=False, swap=True, dims=2):
    """
    Return Shapely Polygon from coordinates.

    - coords: list of alterating latitude / longitude coordinates
    - fix_geom: automatically fix geometry
    """
    assert len(coords) % dims == 0
    number_of_points = int(len(coords) / dims)
    # print(number_of_points)
    coords_as_array = np.array(coords)
    # print(coords_as_array)
    reshaped = coords_as_array.reshape(number_of_points, dims)
    points = [
        (float(i[1]), float(i[0])) if swap else ((float(i[0]), float(i[1])))
        for i in reshaped.tolist()
    ]
    polygon = Polygon(points).buffer(0)
    try:
        assert polygon.is_valid
        return polygon
    except AssertionError:
        if fix_geom:
            return polygon.buffer(0)
        else:
            raise RuntimeError("Geometry is not valid.")


def get_bounding_boxS2(path):
    """
    Function for creating bounding box for multiple S2 tiles (SAFE format), ACOLITE requires
    bounding box for merging products, this finds the minimum bounding box that includes all the
    products

    :param path: path to folder with S2 level 1 products in SAFE format
    :return: bounding box
    """

    s2path = glob(os.path.join(path, 'S2*MSIL1C*'))
    footprint = []
    for s2 in s2path:
        manifest = os.path.join(s2, 'manifest.safe')
        data_object_section = parse(manifest).find("dataObjectSection")
        for data_object in data_object_section:
            if data_object.attrib.get("ID") == "S2_Level-1C_Product_Metadata":
                relpath = os.path.relpath(next(data_object.iter("fileLocation")).attrib["href"])
                abspath = os.path.join(os.path.normpath(s2), relpath)
        prod_meta = parse(abspath)

        i = 0
        for element in prod_meta.iter():
            global_footprint = None
            for global_footprint in element.iter("Global_Footprint"):
                coords = global_footprint.findtext("EXT_POS_LIST").split()
                coords = np.array(coords, dtype='float32')
                if i == 0:
                    footprint.append(_polygon_from_coords(coords))
                i += 1
    maxlon = []
    maxlat = []
    minlon = []
    minlat = []
    for foot in footprint:
        x_, y_ = foot.exterior.coords.xy
        maxlon.append(np.max(x_))
        maxlat.append(np.max(y_))
        minlon.append(np.min(x_))
        minlat.append(np.min(y_))

    bounding_box = (np.min(minlat), np.min(minlon), np.max(maxlat), np.max(maxlon))

    return bounding_box


def create_acolite_settings(inputfolder, output, s2_target_res, merge_tiles=True):
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
    inputfiles = 'inputfile='
    s2paths = glob(inputfolder + '\S2*MSIL1C*')
    for s2 in s2paths:
        inputfiles = inputfiles + s2 + ','

    inputfiles = inputfiles[:-1] + '\n'
    outputfile = 'output=' + output + '\n'
    bounding_box = str(get_bounding_boxS2(inputfolder))[1:-1]
    limit = 'limit=' + bounding_box + '\n'
    l2w = 'l2w_parameters=rhos_*\n'
    target_res = 's2_target_res=' + str(s2_target_res) + '\n'
    merge = 'merge_tiles=' + str(merge_tiles)
    settings = inputfiles + outputfile + limit + l2w + target_res + merge

    with open("acolite_settings.txt", 'w') as dst:
        dst.write(settings)

    return settings

cwd = os.getcwd()
path = r"E:\MastersProjectData\SIT_S2\20190311_103537_103837_slstr_tti-color_500"
dest = os.path.join(path, 'merged_ACOLITE')
if not os.path.exists(dest):
    os.mkdir(dest)
settings = create_acolite_settings(path, dest, 60)
settingsfp = os.path.abspath('aculite_settings.txt')
acolitefp=r'acolite_py_win/dist/acolite'
os.chdir(acolitefp)
comm = r'acolite.exe --cli --settings="..\..\..\acolite_settings.txt"'
subprocess.run(comm, shell=True)
os.chdir(cwd)
destination = os.path.join(dest, 'merged.tif')
destBands = os.path.join(dest, 'reproject_bands')



