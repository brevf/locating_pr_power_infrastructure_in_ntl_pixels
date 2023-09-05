from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


def save_as_pickle(a_file, path):
    # There will still be a geometry column but ability to visualize the dataframe as a geopandas dataframe
    #   will be lost.
    print('Saving to pickle file...')
    pd.DataFrame(a_file).to_pickle(path)
    print('Done')
    return None


class VIIRSImage:

    def __init__(self, ls_path):

        stime = time()
        print(f'Instantiating LandsatImage object for {ls_path}.')

        self.path = ls_path
        self.h5 = h5py.File(self.path)
        self.h5_array = np.array(self.h5['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_BRDF-Corrected_NTL'])
        self.mandatory_qf_array = np.array(self.h5['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/Mandatory_Quality_Flag'])
        self.qf_cloud_mask_array = np.array(self.h5['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/QF_Cloud_Mask'])
        self.mask_array = None
        self.scaled_array = None
        self.filtered_array = None

        # Don't need to pad array like the Landsat images

        self.spin_up()

        print(f'LandsatImage object for {self.path} instantiated in {np.around(time() - stime, decimals=2)} seconds.')

    def spin_up(self):

        stime = time()
        print(f'Loading Quality Flag array...')
        # Already Qualtiy Flag arrays as attributes above, so really not loading anything
        print(f'Quality Flag array loaded in {np.around(time() - stime, decimals=2)} seconds.')
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
        # np.where is like an if else statement np.where(<condition>, <if condition true>, <if condition false>)
        # Only going to look at the those places in mask array that are NaNs (i.e. not set to False by a Quality Flag)

        self.filtered_array = np.where(np.isnan(self.mask_array),
                                       self.scaled_array,
                                       np.nan)

        print(f"Filtered Array shape: {self.filtered_array.shape}")

    def apply_factors_offsets_sr(self):

        scale_factor = 0.1
        additive_offset = 0

        self.scaled_array = np.multiply(self.h5_array, scale_factor)

        return np.add(self.scaled_array, additive_offset)

    def make_masks(self):

        assert self.mandatory_qf_array.shape == self.qf_cloud_mask_array.shape

        gf_rows, gf_cols = self.mandatory_qf_array.shape

        self.mask_array = np.full((gf_rows, gf_cols), np.nan)

        it_mandatory_qf_array = np.nditer(self.mandatory_qf_array, flags=['multi_index'])

        it_qf_cloud_mask_array = np.nditer(self.qf_cloud_mask_array, flags=['multi_index'])

        it_h5_array = np.nditer(self.h5_array, flags=['multi_index'])

        for (qf_cloud_mask_value, qf_array_value, h5_array_value) in zip(it_qf_cloud_mask_array, it_mandatory_qf_array, it_h5_array):

            fill, poor_quality = unpack_mandatory_qf(qf_array_value)

            day, cloud, shadow, sea = qf_cloud_mask(qf_cloud_mask_value)

            missed_fill_value = True if h5_array_value == 65_535 else False

            pixel_index = it_mandatory_qf_array.multi_index

            if fill:
                self.mask_array[pixel_index] = True

            # This should catch any fill values not caught by the quality flag array
            if missed_fill_value:
                self.mask_array[pixel_index] = True

            if poor_quality:
                self.mask_array[pixel_index] = True

            if cloud:
                self.mask_array[pixel_index] = True

            if shadow:
                self.mask_array[pixel_index] = True

            if day:
                self.mask_array[pixel_index] = True

            if sea:
                self.mask_array[pixel_index] = True


def unpack_mandatory_qf(qf):

    fill = False
    if qf == 225:
        fill = True

    poor_quality = False
    if qf == 2 or qf == 3:
        poor_quality = True

    return fill, poor_quality


def qf_cloud_mask(qf):

    unpacked = bin(qf)[2:]

    while len(unpacked) < 16:
        unpacked = '0' + unpacked

    reversed = ''

    unpacked_list = list(unpacked)
    while unpacked_list:
        reversed += unpacked_list.pop()

    day = False
    if reversed[0] == '1':
        day = True

    cloud = False
    if reversed[9] == '1':
        cloud = True

    shadow = False
    if reversed[8] == '1':
        shadow = True

    # looking at bit 1-3, recall that python is not in-point inclusive
    sea = False
    if reversed[1:4] == "011":
        sea = True

    return day, cloud, shadow, sea


def get_diff_array(ls_one, ls_two):

    return np.subtract(ls_one.filtered_array,
                       ls_two.filtered_array,
                       where=~np.isnan(ls_one.filtered_array) & ~np.isnan(ls_two.filtered_array),
                       out=np.full((ls_one.filtered_array.shape[0],
                                    ls_one.filtered_array.shape[1]),
                                   np.nan))


def plot_image_main(ls_one):

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["jet"]
    cmap.set_bad('k')
    my_cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)

    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(ls_one.filtered_array, cmap=my_cmap.cmap)

    plt.show()


def plot_images_diff_main(path_one, path_two):

    # Create LandsatImage objects
    ls_one = VIIRSImage(path_one)
    ls_two = VIIRSImage(path_two)

    # Get the difference between the arrays
    diff_arr = get_diff_array(ls_one, ls_two)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["jet"]
    cmap.set_bad('k')
    my_cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)

    ax = fig.add_subplot(1, 3, 1)

    ax.imshow(ls_one.filtered_array, cmap=my_cmap.cmap)
    ax.set_title('Before')

    ax = fig.add_subplot(1, 3, 2)

    ax.imshow(ls_two.filtered_array, cmap=my_cmap.cmap)
    ax.set_title('After')

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = mpl.colormaps["seismic"]
    cmap.set_bad('k')
    my_cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Difference')

    ax.imshow(diff_arr, cmap=my_cmap.cmap)

    plt.show()


# Convert degrees and decimal minutes to decimal degrees
def dms_to_decdegs(degrees, dec_mins=0, dec_sec=0):
    # Return the decimal degrees
    return degrees + (dec_mins / 60) + (dec_sec / 3600)


# Get a VNP46 grid tile and pixel from a latitude (Coordinate object)
def get_tile_pixel_from_lat(coordinate, south_boundary=False):
    # Tiles run from 0-17 for +90 to -90 latitude
    # Tiles run from 0 - 43199 pixels
    tile_number = 18 - ((coordinate + 90) / 10)
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
    tile_number = ((coordinate + 180) / 10)
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


def bresenham_pixel_approx_of_line(x1, y1, x2, y2):
    """
    This function is from https://babavoss.pythonanywhere.com/python/bresenham-line-drawing-algorithm-implemented-in-py
    It is a bresenham line drawing algorithm that gets the pixels that approximate a straight line for lines
    with any kind of slope (negative, positive, absolute value greater between zero and 1 or greater than one).

    For the same two input points, may get different output points depending on which input point you chose
        to be (x1, y1) and which you chose to be (x2, y2).
    For that reason, we want to run this function with both combinations of the two points and take the
        set union of the points it returns

    This function can handle horizontal lines but not vertical lines because of division by zero

    """

    flip_x_and_y = False

    x, y = x1, y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    gradient = dy / float(dx)

    if gradient > 1:
        # Do the following so algorithm will iterate by 1 unit increments over the axis that does not dominate
        #   the numerator (i.e. if numerator of gradient greater than one, will iterate over 1 by 1 over y-axis
        #   instead of the x-axis)

        flip_x_and_y = True

        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2 * dy - dx
    # Initialize the plotting points

    coordinates = [(x, y)]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1

        coordinates.append((x, y))

    if flip_x_and_y:
        coordinates = [(point[1], point[0]) for point in coordinates]

    return coordinates


def find_pixels_along_line_segment(pixel_one, pixel_two):

    pixels_along_line_segment = ()

    if pixel_one[0] == pixel_two[0]:

        # When points are exactly the same, just return the point that they both equal

        if pixel_one[1] != pixel_two[1]:
            # When dealing with vertical line

            assert pixel_one[0] == pixel_two[0]

            pixels_along_line_segment = tuple(
                [(pixel_one[0], i) for i in
                 range(min([pixel_one[1], pixel_two[1]]), max([pixel_one[1], pixel_two[1]]) + 1)]
            )

        elif pixel_one[1] == pixel_two[1]:

            assert pixel_one == pixel_two

            pixels_along_line_segment = (pixel_one,)

    elif pixel_one[0] != pixel_two[0]:

        pixels_visited_starting_with_pixel_one = bresenham_pixel_approx_of_line(
            x1=pixel_one[0],
            y1=pixel_one[1],
            x2=pixel_two[0],
            y2=pixel_two[1]
        )

        pixels_visited_starting_with_pixel_two = bresenham_pixel_approx_of_line(
            x1=pixel_two[0],
            y1=pixel_two[1],
            x2=pixel_one[0],
            y2=pixel_one[1]
        )

        pixels_along_line_segment = tuple(
            sorted(
                set(pixels_visited_starting_with_pixel_one).union(
                    set(pixels_visited_starting_with_pixel_two)
                )
            )
        )

    return pixels_along_line_segment


class LineGeometry:

    def __init__(self, this_line_geometry, array, array_tile):

        self.vertexes = tuple(zip(this_line_geometry.xy[0], this_line_geometry.xy[1]))

        # print(self.vertexes)

        self.vertex_tiles = ()

        self.vertex_pixels = ()

        self.get_tiles_and_pixels()

        self.pixel_line_segments = []

        self.line_segments_contained_in_tile = []

        self.get_pixels_along_line_segments()

        self.array_values_along_line_segments = []

        self.extract_pixel_values_to_line_segments(array=array, array_tile=array_tile)

    def get_tiles_and_pixels(self):

        all_tiles = []

        all_pixels = []

        for this_vertex in self.vertexes:

            tile = [0, 0]

            pixel = [0, 0]

            # horizontal axis
            tile[0], pixel[0] = get_tile_pixel_from_long(this_vertex[0])

            # vertical axis
            tile[1], pixel[1] = get_tile_pixel_from_lat(this_vertex[1])

            all_tiles.append(tuple(tile))

            all_pixels.append(tuple(pixel))

        # If lines are in more than one tile, will want to explode by tile and take only those
        #   verticies and line segments continaed in the tile you want to keep
        self.vertex_tiles = tuple(all_tiles)

        self.vertex_pixels = tuple(all_pixels)

    def get_pixels_along_line_segments(self):

        for i in range(len(self.vertex_pixels) - 1):

            self.pixel_line_segments.append(
                find_pixels_along_line_segment(
                    pixel_one=self.vertex_pixels[0], pixel_two=self.vertex_pixels[i+1]
                )
            )

        self.pixel_line_segments = tuple(self.pixel_line_segments)

    def extract_pixel_values_to_line_segments(self, array, array_tile):

        for i in range(len(self.pixel_line_segments)):

            if self.vertex_tiles[i] == array_tile and self.vertex_tiles[i + 1] == array_tile:

                # This will keep track of whether a line segment is completely contained with the array_tile
                self.line_segments_contained_in_tile.append(True)

                array_values_along_this_line_segment = []

                for point_pixel in self.pixel_line_segments[i]:

                    array_values_along_this_line_segment.append(array[point_pixel[1], point_pixel[0]])

                self.array_values_along_line_segments.append(tuple(array_values_along_this_line_segment))

            elif not (self.vertex_tiles[i] == array_tile and self.vertex_tiles[i + 1] == array_tile):

                self.line_segments_contained_in_tile.append(False)

                self.array_values_along_line_segments.append(tuple([np.nan] * len(self.pixel_line_segments[i])))

        self.line_segments_contained_in_tile = tuple(self.line_segments_contained_in_tile)

        self.array_values_along_line_segments = tuple(self.array_values_along_line_segments)


def subset_line_gp_df_by_tile(line_gp_df_with_extractions, in_tile=True):

    line_gp_df_with_extractions_exploded = line_gp_df_with_extractions.explode('line_segments_contained_in_tile')

    return line_gp_df_with_extractions_exploded.loc[
        line_gp_df_with_extractions_exploded['line_segments_contained_in_tile'] == in_tile
    ].drop_duplicates()


def extract_pixel_values_to_lines(line_gp_df, array, array_tile=(11, 7)):

    geometry = gpd.GeoSeries(line_gp_df['geometry'])

    # Shapely geometry types: https://shapely.readthedocs.io/en/stable/geometry.html

    if sum(geometry.geometry.type == 'LineString') != len(geometry.geometry.type == 'LineString'):
        # Don't put in try-except block if you want the error to fail loudly
        raise ValueError('Not all feature geometries are LineString geometries.')

    # List of crs attributes: https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS

    # NAD83 and WGS84 Respectively
    if str(geometry.crs) != "EPSG:4269" and str(geometry.crs) !="EPSG:4326":

        print(f"\nVector file has the following Projected Coordinate System or incompatible Geographic Coordinate System : \n{str(geometry.crs)}\n{str(geometry.crs.name)}\n")

        if "WGS" in str(geometry.crs.datum):

            print("Extraction of image pixels will be done using the vector geometry's transformation to EPSG:4326 (WGS84) Geographic Coordinate System decimal degrees.\n")

            geometry = geometry.to_crs("EPSG:4326")

        elif "WGS" not in str(geometry.crs.datum):

            print("Extraction of image pixels will be done using the vector geometry's transformation to EPSG:4269 (NAD83) Geographic Coordinate System decimal degrees.\n")

            geometry = geometry.to_crs("EPSG:4269")

    # print(list(pp_gdf_geometry.to_dict().values())[0].xy[0])

    geometry_as_dict = geometry.to_dict()

    extractions = {
        "index": list(geometry_as_dict.keys()),
        "vertex_tiles": [],
        "vertex_pixels": [],
        "line_segments_contained_in_tile": [],
        "pixel_line_segments": [],
        "array_values_along_line_segments": []
    }

    for this_line in geometry_as_dict.values():

        this_line_geometry = LineGeometry(this_line_geometry=this_line, array=array, array_tile=array_tile)

        extractions["vertex_tiles"].append(this_line_geometry.vertex_tiles)

        extractions["vertex_pixels"].append(this_line_geometry.vertex_pixels)

        extractions["line_segments_contained_in_tile"].append(this_line_geometry.line_segments_contained_in_tile)

        extractions["pixel_line_segments"].append(this_line_geometry.pixel_line_segments)

        extractions["array_values_along_line_segments"].append(this_line_geometry.array_values_along_line_segments)

    assert len(extractions["index"]) == len(extractions["vertex_tiles"]) == len(extractions["vertex_pixels"]) == \
           len(extractions["line_segments_contained_in_tile"]) == len(extractions["pixel_line_segments"]) == \
           len(extractions["array_values_along_line_segments"]) == line_gp_df.shape[0]

    # Returns a new geopandas dataframe separate from the input one with the extractions merged in

    print("\nArray value extract to geometry complete.")

    return subset_line_gp_df_by_tile(
        line_gp_df_with_extractions=line_gp_df.reset_index().merge(
            pd.DataFrame(extractions), left_on='index', right_on='index', how="left"
        ),
        in_tile=True
    )


if __name__ == '__main__':

    if True:

        # Use Path function from pathlib module to avoid having to convert between Windows formatted paths and Linux/Mac formatted paths
        # Put r at front of Windows format path to avoid python confusing backslashes in the path as backslash characters

        OUTPUT_PATH = Path(
            r'C:\Users\brevi\Documents\All_Files\2022_2024_Non_Harvard\Work_with_Ian_Paynter\Locating_PR_Power_Infrastructure_in_NTL_Pixels\outputs')

        test_file1 = Path('/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017262.h11v07.001.2020300190702.h5')

        test_file2 = Path('/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017263.h11v07.001.2021117133537.h5')

        # plot_images_diff_main(path_one=test_file1, path_two=test_file2)

        test_viirs_image = VIIRSImage(test_file1)

        # plot_image_main(ls_one=test_viirs_image)

        transmission_lines_filepath = Path('/Users/brevi/Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Transmission_and_Distribution/g37_electric_lineas_transmision_2014/g37_electric_lineas_transmision_2014.shp')

        # open in geopandas
        tl_gdf = gpd.read_file(transmission_lines_filepath)

        test_tl_gdf_with_extraction_df = extract_pixel_values_to_lines(line_gp_df=tl_gdf, array=test_viirs_image.filtered_array, array_tile=(11, 7))

        # print(test_tl_gdf_with_extraction_df)

        # Can't save geopandas file to shapefile b/c some dataframe elements are tuples or lists
        # Need to summarize these elements before writing to file or convert to pandas dataframe
        #   and save as pickle file
        #   pd.DataFrame(test_tl_gdf_with_extraction_df).to_pickle('./outputs/test_tl_gdf_with_extraction.pkl')

        # Saving Geopandas file
        # https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html
        # test_tl_gdf_with_extraction_df.to_file('./outputs/test_tl_gdf_with_extraction.shp')
        # test_tl_gdf_with_extraction_df.to_file('./outputs/test_tl_gdf_with_extraction.json', driver="GeoJSON")

    if False:

        # There will still be a geometry column but ability to visualize the dataframe as a geopandas dataframe
        #   will be lost.

        # Don't include slashes when use Path() to join the folder and file names

        save_as_pickle(
            a_file=test_tl_gdf_with_extraction_df,
            path=Path(OUTPUT_PATH, "test_tl_gdf_with_extraction_df.pkl")
         )

