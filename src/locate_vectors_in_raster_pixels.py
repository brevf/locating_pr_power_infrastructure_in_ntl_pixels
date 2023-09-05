import sys
# print('geopandas' in sys.modules)
from time import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
# to convert long / lat to points and do calculations on geometries
from shapely import Point


# Convert degrees and decimal minutes to decimal degrees
def dms_to_decdegs(degrees, dec_mins=0, dec_sec=0):
    # Return the decimal degrees
    return degrees + (dec_mins / 60) + (dec_sec / 3600)


class Coordinate:

    # info on slots: https://www.geeksforgeeks.org/python-use-of-__slots__/
    __slots__ = ['dec_degs']

    def __init__(self, coordinate, dms=False):

        coordinate = [coordinate]

        if dms:

            # If the format is Degrees and Decimal Minutes (ddm or DDM)
            if len(coordinate) == 2:
                # Convert to decimal degrees
                self.dec_degs = dms_to_decdegs(coordinate[0], coordinate[1])
            # Otherwise, if the format is Degrees, Minutes, Seconds (dms)
            elif len(coordinate) == 3:
                # Convert to decimal degrees
                self.dec_degs = dms_to_decdegs(coordinate[0], coordinate[1], coordinate[2])
            # Otherwise, it's decimal degrees already
            else:
                self.dec_degs = coordinate[0]

        elif not dms:

            self.dec_degs = coordinate[0]


def get_tile_pixel_from_lat(coordinate, south_boundary=False):
    # Tiles run from 0-17 for +90 to -90 latitude
    # Tiles run from 0 - 43199 pixels
    tile_number = 18 - ((coordinate.dec_degs + 90) / 10)
    # If the tile number is right on a boundary
    if tile_number % 1 == 0:
        # If it's a South boundary
        if south_boundary is True:
            # Subtract 1 from the tile number
            tile_number -= 1
    # print(tile_number)
    # Get the pixel number
    pixel_number = (tile_number % 1) / (1 / 2400)
    # If the pixel number is right on a boundary
    if pixel_number % 1 == 0:
        # If it's a South boundary
        if south_boundary is True:
            # Subtract 1 from the pixel number
            pixel_number -= 1
            # If the pixel number is -1 (previous tile)
            if pixel_number == -1:
                # Change to 2399 (last in previous tile)
                pixel_number = 2399
    return int(np.floor(tile_number)), int(np.floor(pixel_number))

    # Get a VNP46 grid tile and pixel from a longitude (Coordinate object)


def get_tile_pixel_from_long(coordinate, east_boundary=False):
    # Tiles run from 0-35 for -180 to +180 longitude
    tile_number = ((coordinate.dec_degs + 180) / 10)
    # If the tile number is right on a boundary
    if tile_number % 1 == 0:
        # If it's an East boundary
        if east_boundary is True:
            # Subtract 1 from the tile number
            tile_number -= 1
    # print(tile_number)
    # Get the pixel number
    pixel_number = (tile_number % 1) / (1 / 2400)
    # print(pixel_number)
    # If the pixel number is right on a boundary
    if pixel_number % 1 == 0:
        # If it's an East boundary
        if east_boundary is True:
            # Subtract 1 from the pixel number
            pixel_number -= 1
            # If the pixel number is -1 (previous tile)
            if pixel_number == -1:
                # Change to 2399 (last in previous tile)
                pixel_number = 2399
    return int(np.floor(tile_number)), int(np.floor(pixel_number))


def linestring_to_points(line):
    return list(line.coords)


class Vector:

    # need to transform to geographic coordinate system
    def __init__(self, vec, vec_type="point", transform=""):

        # Get a VNP46 grid tile and pixel from a latitude (Coordinate object)

        self.type = vec_type.strip(" ").lower()

        if transform:
            vec = vec.to_crs(transform)

        self.crs = vec.crs

        if self.type == "point":

            vec['gdf_id'] = vec.index

            self.Longitude = vec.Longitude

            self.Latitude = vec.Latitude

        elif self.type == "line":

            vec = vec.explode(index_parts=True)

            # only want the first level index
            vec['gdf_id'] = vec.index.get_level_values(0)

            vec['point'] = vec.apply(lambda l: linestring_to_points(l['geometry']), axis=1)

            vec = vec.drop(['geometry'], axis=1)

            vec = vec.explode('point').reset_index(drop=True)

            self.Longitude = list(vec.apply(lambda l: l['point'][0], axis=1))

            self.Latitude = list(vec.apply(lambda l: l['point'][1], axis=1))

            vec['point'] = vec.apply(lambda l: Point(l['point']), axis=1)

            vec = gpd.GeoDataFrame(vec, crs=self.crs, geometry=vec['point']).drop(['point'], axis=1)

        elif self.type == "polygon":
            pass

        self.point = vec

    def to_pt_with_tile_and_pixel(self):

        self.point['vtile'] = [get_tile_pixel_from_lat(Coordinate(l))[0] for l in self.Latitude]
        self.point['htile'] = [get_tile_pixel_from_long(Coordinate(l))[0] for l in self.Longitude]
        # Grab pixel numbers
        self.point['vpix'] = [get_tile_pixel_from_lat(Coordinate(l))[1] for l in self.Latitude]
        self.point['hpix'] = [get_tile_pixel_from_long(Coordinate(l))[1] for l in self.Longitude]

        return self.point[['gdf_id', 'vtile', 'htile', 'vpix', 'hpix']].drop_duplicates()





class ViirsNtlImage:

    def __init__(self, ls_path):

        stime = time()
        print(f'Instantiating VIIRS NTL Image object for {ls_path}.')

        self.path = ls_path
        self.data_array = np.array(
            h5py.File(self.path)['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['DNB_BRDF-Corrected_NTL']
        )
        self.qf_array = np.array(
            h5py.File(self.path)['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['Mandatory_Quality_Flag']
        )
        self.cloud_mask = np.array(
            h5py.File(self.path)['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['QF_Cloud_Mask']
        )

        self.mask_array = None
        self.scaled_array = None
        self.filtered_array = None

        self.spin_up()

        print(f'VIIRS NTL Image object for {self.path} instantiated in {np.around(time() - stime, decimals=2)} seconds.')

    def spin_up(self):

        print(f'Making mask from Quality Flags to filter array...')
        ctime = time()
        self.make_masks()
        print(f'Mask made in {np.around(time() - ctime, decimals=2)} seconds.')
        print(f'Applying Scale Factor and Additive Offset...')
        ctime = time()
        self.apply_factors_offsets_sr()
        print(f'Scale factors and additive offsets applied in {np.around(time() - ctime, decimals=2)} seconds.')
        print(f'Filtering array using mask...')
        ctime = time()
        #self.filtered_array = np.empty((self.tif.asarray().shape[0], self.tif.asarray().shape[1]))
        self.get_filtered_array()
        print(f'Array filtered using mask in {np.around(time() - ctime, decimals=2)} seconds.')

    def get_filtered_array(self):

        self.filtered_array = np.where(np.invert(self.mask_array),
                                       self.scaled_array,
                                       np.nan)

        print(f"Filtered Array shape: {self.filtered_array.shape}")

    def apply_factors_offsets_sr(self):

        scale_factor = 0.1
        additive_offset = 0.0

        self.scaled_array = np.multiply(self.data_array, scale_factor)

        return np.add(self.scaled_array, additive_offset)

    def make_masks(self):

        gf_rows, gf_cols = self.qf_array.shape

        self.mask_array = np.full((gf_rows, gf_cols), False)

        it_cloud_mask = np.nditer(self.cloud_mask, flags=['multi_index'])

        it_qf_array = np.nditer(self.qf_array, flags=['multi_index'])

        # it_cloud_mask and it_qf_array should be same number of iterations b/c the arrays are the same size
        #   but these objects don't have length attributes so can't check (maybe can convert to list and check)

        for (cloud_mask_value, qf_array_value) in zip(it_cloud_mask, it_qf_array):

            fill, cloud, shadow = unpack_qf(cloud_mask_value, qf_array_value)

            pixel_index = it_cloud_mask.multi_index

            if fill:
                self.mask_array[pixel_index] = True

            if cloud:
                self.mask_array[pixel_index] = True

            if shadow:
                self.mask_array[pixel_index] = True

            # if water:
            #     self.mask_array[pixel_index] = True


def unpack_qf(cloud_mask, qf):

    unpacked = bin(cloud_mask)[2:]

    while len(unpacked) < 16:
        unpacked = '0' + unpacked

    reversed = ''

    unpacked_list = list(unpacked)
    while unpacked_list:
        reversed += unpacked_list.pop()

    fill = False
    if qf == 255:
        fill = True

    # this only for cirrus clouds; there is separate flag for cloud / clear confidence
    cloud = False
    if reversed[9] == '1':
        cloud = True

    shadow = False
    if reversed[8] == '1':
        shadow = True

    # removed water because may be interested in economic activity near water
    # also some places post hurricane may be classified as water and don't want to remove these

    # water = False
    # if reversed[7] == '1':
    #     water = True

    return fill, cloud, shadow


def extract_pixel_values_to_points(points, array, tile=(11, 7)):

    points_in_tile = points[(points['htile'] == tile[0]) & (points['vtile'] == tile[1])]

    rows = list(points_in_tile['vpix'])

    columns = list(points_in_tile['hpix'])

    # ask if pixel numbers start at zero

    extract = {"vpix": rows, "hpix": columns, "values": list(array[rows, columns])}

    extract_df = pd.DataFrame.from_dict(extract)

    points_to_pixel_values = pd.merge(
        points_in_tile, extract_df, on=['vpix', 'hpix'], how='left', sort=True, validate="m:m"
    )

    # vpix and hpix not unique in left dataset

    return points_to_pixel_values


def extract_pixel_values_to_lines(lines, array, tile=(11, 7)):

    lines_to_points = Vector(vec=lines, vec_type="line", transform="EPSG:4269").point

    extract_pixel_values_to_points(points=lines_to_points, array=array, tile=tile)

    # see following stackexchange post for more on how to reduce lines/polygons to their constituent points
    # https://gis.stackexchange.com/questions/238533/extracting-points-from-linestring-or-polygon-and-making-dictionary-out-of-them-i

    # need to break-up multipoint geometries
    # see following link for more info: https://gis.stackexchange.com/questions/378806/multi-part-geometries-do-not-provide-a-coordinate-sequence-error-when-extracti
    # need to break each line into its associated points and treat each record per line like would point record

    pass

def extract_pixel_values_to_polygons():

    # need to find the smallest rectangle of pixels containing all the verticies of the polygon
    # associate to this polygon the value of each pixel in this rectangle

    pass



if True:

    def main(path_one):

        # Create LandsatImage objects
        ls_one = ViirsNtlImage(path_one)


        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.colormaps["jet"]
        cmap.set_bad('k')
        my_cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        fig = plt.figure(figsize=(12, 12), constrained_layout=True)

        ax = fig.add_subplot(1, 3, 1)

        ax.imshow(ls_one.filtered_array, cmap=my_cmap.cmap)
        ax.set_title('NTL')

        plt.show()

# The date the observation was taken is given in the file name as A2017262, the Julien Date year 2017, day 262

# VNP46A2_h11v07_2017_09_19 = h5py.File(test_file, 'r')
#
# print(VNP46A2_h11v07_2017_09_19['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['Mandatory_Quality_Flag'])

# print(VNP46A2_h11v07_2017_09_19['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields'].keys())


# Todo: ask Ian / Peter about how to account for fact that grid is linear even though Earth is round
#   Ask if visuzlizing the VIIRS NTL data correctly
#   Ask Peter if horizontal and vertical pixel indicies should start at zero

# print(VNP46A2_h11v07_2017_09_19['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['Mandatory_Quality_Flag'])
# print(VNP46A2_h11v07_2017_09_19['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['QF_Cloud_Mask'])
# print(VNP46A2_h11v07_2017_09_19['HDFEOS']["GRIDS"]['VNP_Grid_DNB']['Data Fields']['Snow_Flag'])

# The NTL h5 file

if False:

    test_file1 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017262.h11v07.001.2020300190702.h5'
    #
    # test_file2 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017263.h11v07.001.2021117133537.h5'
    #
    main(path_one=test_file1)

    ls_one = ViirsNtlImage(ls_path=test_file1)



# Point at shapefile

power_plants_filepath = '/Users/brevi/Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Production/Power_Plants_US_2022/Power_Plants.shp'

transmission_lines_filepath = '/Users/brevi\Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Transmission_and_Distribution/g37_electric_lineas_transmision_2014/g37_electric_lineas_transmision_2014.shp'

# Power Plants

if False:
    pp_gdf = gpd.read_file(Path(power_plants_filepath))

    pp_gdf_points = Vector(vec=pp_gdf, vec_type="point", transform="EPSG:4269").to_pt_with_tile_and_pixel()


    pp_gdf_points_to_pixel = extract_pixel_values_to_points(points=pp_gdf_points, array=ls_one.filtered_array, tile=(11, 7))

    print(pp_gdf_points_to_pixel.head(n=25))


    # the geometry column won't display the long / lat to its full precison,
    #   so there may look like there are duplicate geometries when there are not

    # fig, ax = plt.subplots()
    # tl_gdf.plot(ax=ax)
    # plt.show()




# transmission lines
if True:
    tl_gdf = gpd.read_file(Path(transmission_lines_filepath))

    print(tl_gdf.loc[0, "geometry"])

    exit()


    # Todo: Use the native geographic coordinate system when finding tile and pixels
    #  But Transform all shapefiles when display together


    # Todo: Make sure the longitude and latitude are being read in correctly with the Coordinates class
    #   convert to lat long and/or make sure that don't convert to decimal degrees if already in decimal degrees
    #   Make sure the line function works then start on the polygon function


    # Todo: Merge pixel values to lines and think about how to handle polygons

    tl_gdf_points = Vector(vec=tl_gdf, vec_type="line", transform="EPSG:4269").to_pt_with_tile_and_pixel()


    tl_gdf_points_to_pixel = extract_pixel_values_to_points(points=tl_gdf_points, array=ls_one.filtered_array, tile=(11, 7))

    print(tl_gdf_points_to_pixel.head(n=50))

    fig, ax = plt.subplots()
    plt.scatter(tl_gdf_points_to_pixel.vpix, tl_gdf_points_to_pixel.hpix, c=tl_gdf_points_to_pixel.values, alpha=0.3)
    ax.axis('equal')
    plt.show()

    exit()

    tl_gdf['gdf_id'] = tl_gdf.index

    print(pd.merge(
            tl_gdf, tl_gdf_points_to_pixel, on=['gdf_id'], how='left', sort=True, validate="1:m"
        ).tail(n=25)
    )


    # the geometry column won't display the long / lat to its full precison,
    #   so there may look like there are duplicate geometries when there are not

    # fig, ax = plt.subplots()
    # tl_gdf.plot(ax=ax)
    # plt.show()



# Todo: The idea was for you to take a crack at matching the infrastructure data to the VIIRS pixels,
#   following the same workflow that we laid out in our last meeting.
#   If you want to go a bit further with it, brainstorm how we could summarize the infrastructure data by pixel
#   (i.e. number of substations per pixel, length of powerlines in the pixel, etc.).
#   You could even try to implement some of those metrics, if you get a chance, and we can expand on them when we meet.

