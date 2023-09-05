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

if __name__ == '__main__':

    if False:

        test_file1 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017262.h11v07.001.2020300190702.h5'

        test_file2 = '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017263.h11v07.001.2021117133537.h5'

        plot_images_main(path_one=test_file1, path_two=test_file2)

        # ls_one = ViirsNtlImage(ls_path=test_file1)
