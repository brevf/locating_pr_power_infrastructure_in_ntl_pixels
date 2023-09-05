from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


class VIIRSImageLite:

    # This lite version of VIIRSImage Class just gets the class's attributes without creating anything else

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

        print(
            f'LandsatImage object for {self.path} instantiated in {np.around(time() - stime, decimals=2)} seconds.')


class InfrastructurePixels:

    def __init__(self, template_array):
        print(f'Instantiating LandsatImage object for {ls_path}.')

        self.infrastructure_pixels_array = None

        make_blank_array(template_array=template_array)

    def make_blank_array(self, template_array):

        gf_rows, gf_cols = template_array.shape

        self.infrastructure_pixels_array = np.full((gf_rows, gf_cols), 0)


def make_blank_array(template_array):

    gf_rows, gf_cols = template_array.shape

    blank_array = np.full((gf_rows, gf_cols), 0)

    return blank_array


def plot_array(array):

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["jet"]
    cmap.set_bad('k')
    my_cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)

    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(array, cmap=my_cmap.cmap)

    plt.show()

    return None


if __name__ == '__main__':

    OUTPUT_PATH = Path(
        r'C:\Users\brevi\Documents\All_Files\2022_2024_Non_Harvard\Work_with_Ian_Paynter\Locating_PR_Power_Infrastructure_in_NTL_Pixels\outputs')

    test_file1 = Path(
        '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017262.h11v07.001.2020300190702.h5')

    test_file2 = Path(
        '/Users/brevi/Documents/All_Files/2022_2024_Non_Harvard/Work_with_Ian_Paynter/LAADS_Tools/inputs/2017_09_19_to_2017_09_20_VNP46A2_5000_h11v07_pr_maria/VNP46A2.A2017263.h11v07.001.2021117133537.h5')

    test_viirs_image = VIIRSImageLite(ls_path=test_file1)

    test_blank_array = make_blank_array(template_array=test_viirs_image.h5_array)

    pp_gdf_with_extractions_df = pd.read_pickle(Path(OUTPUT_PATH, "pp_gdf_with_extractions_df.pkl"))

    # see if can join this sub dataframe back to original dataframe using point tiles and point pixels or
    #   at least find some way to look up what pieces of infrastructure are in a given tile and pixel combination

    pp_gdf_with_extractions_sub_df = pp_gdf_with_extractions_df[
        ['point_tile', 'point_pixel', 'array_value_at_pixel']
    ].drop_duplicates()

    # Will need to explode dataframe at this step for lines and polygons

    pp_gdf_with_extractions_sub_df_in_correct_tile = pp_gdf_with_extractions_sub_df.loc[
        pp_gdf_with_extractions_sub_df['point_tile'] == (11, 7)
    ]

    for point_pixel in pp_gdf_with_extractions_sub_df_in_correct_tile['point_pixel'].tolist():

        test_blank_array[point_pixel[1], point_pixel[0]] = 1

    plot_array(array=test_blank_array)













