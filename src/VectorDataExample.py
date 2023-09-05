import sys
# print('geopandas' in sys.modules)
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt


class Coordinate:

    # info on slots: https://www.geeksforgeeks.org/python-use-of-__slots__/
    __slots__ = ['dec_degs']

    def __init__(self, coordinate):

        coordinate = [coordinate]

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


# Convert degrees and decimal minutes to decimal degrees
def dms_to_decdegs(degrees, dec_mins=0, dec_sec=0):
    # Return the decimal degrees
    return degrees + (dec_mins / 60) + (dec_sec / 3600)


# Get a VNP46 grid tile and pixel from a latitude (Coordinate object)
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

# Point at shapefile
f = Path('/Users/brevi/Documents/All_Files/Harvard_GSAS_RSI/Senior_Thesis/Data/Infrastructure/Electricity_Production/Power_Plants_US_2022/Power_Plants.shp')

# open in geopandas
pp_gdf = gpd.read_file(f)


# Print first 5 lines and plot whole dataframe
# print(pp_gdf.head())

# fig, ax = plt.subplots()
# pp_gdf.plot(ax=ax)
# plt.show()

# Filter to only PR, using VIIRS tile extents

# Use Ian's functions to get VIIRS tiles & pixels
# Grab tile numbers
pp_gdf['vtile'] = [get_tile_pixel_from_lat(Coordinate(l))[0] for l in pp_gdf.Latitude]
pp_gdf['htile'] = [get_tile_pixel_from_long(Coordinate(l))[0] for l in pp_gdf.Longitude]
# Grab pixel numbers
pp_gdf['vpix'] = [get_tile_pixel_from_lat(Coordinate(l))[1] for l in pp_gdf.Latitude]
pp_gdf['hpix'] = [get_tile_pixel_from_long(Coordinate(l))[1] for l in pp_gdf.Longitude]

print(pp_gdf.head())

# Pick out PR tile
pp_gdf_pr = pp_gdf[((pp_gdf.vtile == 7) & (pp_gdf.htile == 11))]

# Plot
fig, ax = plt.subplots()
plt.scatter(pp_gdf_pr.Longitude, pp_gdf_pr.Latitude)
ax.axis('equal')
plt.show()


