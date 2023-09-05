import tifffile
import numpy as np
import matplotlib as mpl
from time import time
from pathlib import Path
from matplotlib import pyplot as plt


class LandsatImage:

    def __init__(self, ls_path):

        stime = time()
        print(f'Instantiating LandsatImage object for {ls_path}.')

        self.path = ls_path
        self.tif = tifffile.TiffFile(self.path)
        self.qf_array = None
        self.mask_array = None
        self.scaled_array = None
        self.filtered_array = None

        self.pixel_width = self.tif.pages[0].tags['ModelPixelScaleTag'].value[0]
        self.left_edge = self.tif.pages[0].tags['ModelTiepointTag'].value[3] - (self.pixel_width / 2)
        self.columns = int(self.tif.pages[0].tags['ImageWidth'].value)
        self.rows = int(self.tif.pages[0].tags['ImageLength'].value)
        self.right_edge = self.left_edge + (self.tif.pages[0].tags['ModelPixelScaleTag'].value[0] * self.columns)

        self.cols_to_left = None
        self.cols_to_right = None

        self.spin_up()

        print(f'LandsatImage object for {self.path} instantiated in {np.around(time() - stime, decimals=2)} seconds.')

    def spin_up(self):

        stime = time()
        print(f'Loading Quality Flag array...')
        self.get_qf_array()
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

        self.filtered_array = np.where(np.isnan(self.mask_array),
                                       self.scaled_array,
                                       np.nan)

        print(f"Filtered Array shape: {self.filtered_array.shape}")

    def apply_factors_offsets_sr(self):

        scale_factor = 2.75e-5
        additive_offset = -0.2

        self.scaled_array = np.multiply(self.tif.asarray(), scale_factor)

        return np.add(self.scaled_array, additive_offset)

    def make_masks(self):

        gf_rows, gf_cols = self.qf_array.shape

        self.mask_array = np.full((gf_rows, gf_cols), np.nan)

        it = np.nditer(self.qf_array, flags=['multi_index'])

        for qf_value in it:

            fill, cloud, shadow, water = unpack_qf(qf_value)

            pixel_index = it.multi_index

            if fill:
                self.mask_array[pixel_index] = True

            if cloud:
                self.mask_array[pixel_index] = True

            if shadow:
                self.mask_array[pixel_index] = True

            if water:
                self.mask_array[pixel_index] = True

    def get_qf_array(self):

        split_file = str(self.path).split('_')
        file_suffix = split_file[-2] + '_' + split_file[-1]
        quality_flag_path = Path(str(self.path).replace(file_suffix, 'QA_PIXEL.TIF'))

        self.qf_array = tifffile.imread(str(quality_flag_path))

    def pad_array(self):

        print(f"Filtered Array shape before padding: {self.filtered_array.shape}")

        if self.cols_to_left > 0:
            left_cols = np.full((self.rows, self.cols_to_left), np.nan)
            print(f"Left cols shape: {left_cols.shape}")

            self.filtered_array = np.hstack((left_cols, self.filtered_array))

        if self.cols_to_right > 0:
            right_cols = np.full((self.rows, self.cols_to_right), np.nan)
            print(f"Right cols shape: {right_cols.shape}")
            self.filtered_array = np.hstack((self.filtered_array, right_cols))

        print(f"Filtered Array shape after padding: {self.filtered_array.shape}")


def unpack_qf(qf):

    unpacked = bin(qf)[2:]

    while len(unpacked) < 16:
        unpacked = '0' + unpacked

    reversed = ''

    unpacked_list = list(unpacked)
    while unpacked_list:
        reversed += unpacked_list.pop()

    fill = False
    if reversed[0] == '1':
        fill = True

    cloud = False
    if reversed[6] == '0':
        cloud = True

    shadow = False
    if reversed[4] == '1':
        shadow = True

    water = False
    if reversed[7] == '1':
        water = True

    return fill, cloud, shadow, water


def get_diff_array(ls_one, ls_two):

    pad_arrays(ls_one, ls_two)

    return np.subtract(ls_one.filtered_array,
                       ls_two.filtered_array,
                       where=~np.isnan(ls_one.filtered_array) & ~np.isnan(ls_two.filtered_array),
                       out=np.full((ls_one.filtered_array.shape[0],
                                    ls_one.filtered_array.shape[1]),
                                   np.nan))


def pad_arrays(ls_one, ls_two):

    array_left_edge = ls_one.left_edge

    if ls_two.left_edge < ls_one.left_edge:
        array_left_edge = ls_two.left_edge

    array_right_edge = ls_one.right_edge

    if ls_two.right_edge > ls_one.right_edge:
        array_right_edge = ls_two.right_edge

    ls_one.cols_to_left = int(round((ls_one.left_edge - array_left_edge) / ls_one.pixel_width))
    ls_one.cols_to_right = int(round((array_right_edge - ls_one.right_edge) / ls_one.pixel_width))

    ls_two.cols_to_left = int(round((ls_two.left_edge - array_left_edge) / ls_two.pixel_width))
    ls_two.cols_to_right = int(round((array_right_edge - ls_two.right_edge) / ls_two.pixel_width))

    print(f"LS image one left cols: {ls_one.cols_to_left}, right cols: {ls_one.cols_to_right}")
    print(f"LS image two left cols: {ls_two.cols_to_left}, right cols: {ls_two.cols_to_right}")

    ls_one.pad_array()
    ls_two.pad_array()


def main(path_one, path_two):

    # Create LandsatImage objects
    ls_one = LandsatImage(path_one)
    ls_two = LandsatImage(path_two)

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


if __name__ == "__main__":

    # Path to first image (earliest in time)
    image_one_path = Path(r'F:\Leidos\NASA\HurricaneMaria\LC08_L2SP_005047_20170901_20200903_02_T1_SR_B3.TIF')

    # Path to second image (latest in time)
    image_two_path = Path(r'F:\Leidos\NASA\HurricaneMaria\LC08_L2SP_005047_20171019_20200902_02_T1_SR_B3.TIF')

    main(image_one_path,
         image_two_path)
