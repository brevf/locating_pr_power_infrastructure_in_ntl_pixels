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

        for (qf_cloud_mask_value, qf_array_value) in zip(it_qf_cloud_mask_array, it_mandatory_qf_array):

            fill, poor_quality = unpack_mandatory_qf(qf_array_value)

            day, cloud, shadow, sea = qf_cloud_mask(qf_cloud_mask_value)

            pixel_index = it_mandatory_qf_array.multi_index

            if fill:
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


def plot_images_main(path_one, path_two):

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


class PointGeometry:

    def __init__(self, this_point_geometry, array):

        self.point = tuple(zip(this_point_geometry.xy[0], this_point_geometry.xy[1]))

        # print(self.point)

        self.point_tile = ()

        self.point_pixel = ()

        self.get_tiles_and_pixels()

        self.array_value_at_point = ()

        self.extract_pixel_values_to_points(array=array)

    def get_tiles_and_pixels(self):

        tile = [0, 0]

        pixel = [0, 0]

        # horizontal axis
        tile[0], pixel[0] = get_tile_pixel_from_long(self.point[0])

        # vertical axis
        tile[1], pixel[1] = get_tile_pixel_from_lat(self.point[1])

        self.point_tile = tuple(tile)

        self.point_pixel = tuple(pixel)

    def extract_pixel_values_to_points(self, array):

        # y-coordinate [1] is the row index, x-coordinate [0] is the column index

        self.array_value_at_point = tuple(array[self.point_pixel[1], self.point_pixel[0]])


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

        # When points are exactly the same, will just return empty tuple

        if pixel_one[1] != pixel_two[1]:
            # When dealing with vertical line

            assert pixel_one[0] == pixel_two[0]

            pixels_along_line_segment = tuple(
                [(pixel_one[0], i) for i in
                 range(min([pixel_one[1], pixel_two[1]]), max([pixel_one[1], pixel_two[1]]) + 1)]
            )

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

    def __init__(self, this_line_geometry, array):

        self.vertexes = tuple(zip(this_line_geometry.xy[0], this_line_geometry.xy[1]))

        # print(self.vertexes)

        self.vertex_tiles = ()

        self.vertex_pixels = ()

        self.get_tiles_and_pixels()

        self.pixel_line_segments = []

        self.get_pixels_along_line_segments()

        self.array_values_along_line_segments = ()

        self.extract_pixel_values_to_line_segments(array=array)

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

    def extract_pixel_values_to_line_segments(self, array):

        rows = []

        columns = []

        for this_column, this_row in self.pixel_line_segments:

            rows.append(this_row)

            columns.append(this_column)

        self.array_values_along_line_segments = tuple(array[rows, columns])




if False:

    test_file1 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017262.h11v07.001.2020300190702.h5'

    test_file2 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017263.h11v07.001.2021117133537.h5'

    plot_images_main(path_one=test_file1, path_two=test_file2)

    # ls_one = ViirsNtlImage(ls_path=test_file1)



# Point at shapefile


if True:

    power_plants_filepath = '/Users/brevi/Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Production/Power_Plants_US_2022/Power_Plants.shp'

    transmission_lines_filepath = '/Users/brevi\Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Transmission_and_Distribution/g37_electric_lineas_transmision_2014/g37_electric_lineas_transmision_2014.shp'

    # Power Plants

    # open in geopandas
    tl_gdf = gpd.read_file(transmission_lines_filepath)

    # make a deep copy so don't affect original dataframe

    def extract_pixel_values_to_points(point_gp_df, array):

        geometry = gpd.GeoSeries(point_gp_df['geometry'])

        if str(geometry.crs) != "EPSG:4269":
            print(
                f"Vector file has CRS {str(geometry.crs)}. Extration of image pixels will be done using the vector geometry's transfomation to EPSG:4269 (NAD83) coordinates.")
            geometry = geometry.to_crs("EPSG:4269")

        geometry_as_dict = geometry.to_dict()

        extractions = {
            "index": list(geometry_as_dict.keys()),
            "point_tile": [],
            "point_pixel": [],
            "array_value_at_pixel": []
        }

        for this_point in geometry_as_dict.values():

            this_point_geometry = PointGeometry(this_point_geometry=this_point, array=array)

            extractions["point_tile"].append(this_point_geometry.point_tile)

            extractions["point_pixel"].append(this_point_geometry.point_pixel)

            extractions["array_value_at_pixel"].append(this_point_geometry.array_value_at_point)

        return extractions


    def extract_pixel_values_to_lines(line_gp_df, array):

        geometry = gpd.GeoSeries(line_gp_df['geometry'])

        if str(geometry.crs) != "EPSG:4269":
            print(f"Vector file has CRS {str(geometry.crs)}. Extration of image pixels will be done using the vector geometry's transfomation to EPSG:4269 (NAD83) coordinates.")
            geometry = geometry.to_crs("EPSG:4269")

        # print(list(pp_gdf_geometry.to_dict().values())[0].xy[0])

        geometry_as_dict = geometry.to_dict()

        extractions = {
            "index": list(geometry_as_dict.keys()),
            "vertex_tiles": [],
            "vertex_pixels": [],
            "pixel_line_segments": [],
            "array_values_along_line_segments": []
        }

        for this_line in geometry_as_dict.values():

            this_line_geometry = LineGeometry(this_line_geometry=this_line, array=array)

            extractions["vertex_tiles"].append(this_line_geometry.vertex_tiles)

            extractions["vertex_pixels"].append(this_line_geometry.vertex_pixels)

            extractions["pixel_line_segments"].append(this_line_geometry.pixel_line_segments)

            extractions["array_values_along_line_segments"].append(this_line_geometry.array_values_along_line_segments)

        return extractions

    # Todo: See if workflow for lines works
    #  Figure out how you want to handle the extractions dictionary after converting to pandas dataframe
    #  See if you want to pull out any unnecessary data / select tiles wanted before joining with geo pandas dataframe
    #  Start thinking about how to handle points and lines

    """
    
    Thinking about polygons:
        
        Is there a Bresenham's line algorithm or DDA style approximation of a polygon using pixels
        
        Use Bresenham's line algorithm to find all the pixels along the edges of the polygon
        
        Find a bounding box of the polygon. The draw horizontal line down the bounding box iteratively.
        Any point between the 1st and 2nd or 3rd and 4th, etc. vertexes of the polygon touched by the horizontal line 
        should be inside the polygon (think about this and make sure no edge cases?)
        
        Do this until touch the bottom vertex of the polygon
        
        Also, make sure that polygon doesn't fall entirely inside one pixel
        
    """


if False:

        # get the pixel values at each pixel then get them along the segments

        # skip if both vertexes are in the same pixel (i.e. there is no segment)

        # Looking at one line segment at a time

        # ((1099, 441), (1120, 415))

        # all_pixels[5]

        # all_pixels[6]









        # # ((1099, 441), (1120, 415))
        #
        # print(all_pixels[5], all_pixels[6])
        #
        # # ((441, 1099), (415, 1120))
        #
        # # print((441, 1099), (415, 1120))
        #
        # test = FindPixelsAlongLineSegment(pixel_one=(441, 1010), pixel_two=(441, 900))
        #
        # print(test.pixels_along_line_segment)



        # Todo: Test the line segment class and figure out how to connect back with original data
        #   Make sure you join the pixels each vertex belongs to with the pixels along the line segments
        #   (That is each self.pixels_along_line_segment should be concatenated with somehow with the the list of pixels
        #   that each line vertex belongs to)
        #   Draw out on graph paper how you expect it to work (i.e. what pixels should be returned) before running
        #  If line segment class works, extract pixel values from numpy array
        #  Figure out how to match the vertexes and concave hull of polygons to their associated pixels
        #  Figure out a way to check what type of vector dealing with (point, line, polygon)





                # want to do the same thing as in the above block, up with the north pixel to the east and the south
                #   pixel to the west

                # include some code to handle cases where vertical or horizontal pixel axes are equal

                # try and work more of the above into functions or classes


    """
    
    Thoughts about getting every pixel that a line vector crosses:
    
        I was initially thinking about finding the equation of the straight line between every point that forms the 
        line. To do this I would either need to convert the geometry to a projection that preserves distance and one 
        that preserves direction (angles) so that I would use together to find all the points on a line between any two
        points. 
        I learned it would probably be easier just to work with decimal degree coordinates using geodesic line and 
        somehow sampling a number of points along this line. Geodesic lines allow you to use degree coordinates but
        account for the curvature of the Earth. There ways to do this using SLERP (Spherical Linear Interpolation) and
        modules in python like Geodesic. The difficult with this is finding the how many points to look at between
        the two points that define the line. I considered just using the data type of the numpy array containing the 
        coordinates (e.g., float32, float64, etc.) and finding the smallest number that can be expressed in that 
        data type. Then I could divide the distance between the two points defining the line by this number to get the 
        maximum number of points I could sample along the line.
        
        ###
        
        this_dtype = coord_array.dtype


        num_closest_to_zero_array = np.array([0, 1]).astype(this_dtype)
    
        num_closest_to_zero = np.nextafter(num_closest_to_zero_array[0], num_closest_to_zero_array[1])
    
        print(num_closest_to_zero)
        
        ###
        
        
        However, it would probably been even more computational efficiently to find the grid block that each point
        in the line belongs and determine what blocks you need to pass to get from one containing a point to another 
        (kind of like a Manhattan distance, looking at what "city blocks" you need to pass to get from one city 
        block to another). This is ultimately what I settled on.
        
        Not sure if this is the most accurate way to do things but I should ask Ian if he knows a better way.
        
        Also, you are trying to extract the image values between each line segment and at the vertexes of each line,
        not just at the vertexes. Is this worth doing.
        
        https://gis.stackexchange.com/questions/368740/finding-line-equation-on-earth-using-two-gps-coordinates
        https://stackoverflow.com/questions/27605242/points-on-a-geodesic-line
        https://www.gpxz.io/blog/sampling-points-on-a-line
        
        http://vterrain.org/Misc/distance.html#:~:text=The%20simplest%20way%20to%20calculate,distance%20%3D%20angle%20*%20pi%20*%20radius
        
        https://stackoverflow.com/questions/38477908/smallest-positive-float64-number
        https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
        https://www.geeksforgeeks.org/change-data-type-of-given-numpy-array/
        
        
        2023-07-31
        
        I learned that an even better way to approximate a line with image pixels is by using Bresenham's line algorithm.
        
        https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
        https://babavoss.pythonanywhere.com/python/bresenham-line-drawing-algorithm-implemented-in-py
    
        There are other line line drawing algorithms like DDA, but Bresenham's algortihm seems to be best for our purposes.
        https://www.geeksforgeeks.org/comparions-between-dda-and-bresenham-line-drawing-algorithm/
        https://www.tutorialspoint.com/difference-between-dda-and-bresenham-line-drawing-algorithm
        
        
        
        
        
    """


# Todo: The idea was for you to take a crack at matching the infrastructure data to the VIIRS pixels,
#   following the same workflow that we laid out in our last meeting.
#   If you want to go a bit further with it, brainstorm how we could summarize the infrastructure data by pixel
#   (i.e. number of substations per pixel, length of powerlines in the pixel, etc.).
#   You could even try to implement some of those metrics, if you get a chance, and we can expand on them when we meet.


