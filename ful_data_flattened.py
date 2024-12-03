import numpy as np
import matplotlib.pyplot as plt
from Params import*
from nsOCT_maximum_spatial_periodOCT import *



def full_data_flattened(nsOCT,bins, do_compute):
    Number_of_pixels = Params['Number_of_pixels']
    Full_data_flattened = Params['Full_data_flattened']

    if do_compute:
        three_d_nsOCT = nsOCT
        NaN_limit = Params['threshold_spatial_period']



        pixels_selected = Params['pixels_selected']
        three_d_nsOCT = three_d_nsOCT[:, pixels_selected[0]-1:pixels_selected[1],:]
        print('The dimension of three_d_nsOCT is', np.shape(three_d_nsOCT))

        flat_data = three_d_nsOCT.flatten()
        print('dimension of flat data is', np.shape(flat_data))



        # sns.distplot(flat_data)
        plt.hist(flat_data, bins=bins, range=(histograms_x_axis_range[0], histograms_x_axis_range[1]),density=True,color='g',rwidth=0.8)
        plt.tight_layout()
        plt.xlabel("Spatial period (nm)")
        plt.ylabel("Amplitude (a.u)")
        plt.ylim(0, histograms_y_axis_limit)
        plt.tight_layout()

        script_dir = os.path.dirname(__file__)
        full_flat_dir = os.path.join(script_dir, Full_data_flattened)
        plt.savefig(os.path.join(full_flat_dir, "Full_data_histo.png"))
        # plt.show()

        flat_data[flat_data < NaN_limit] = np.NaN

        full_data_mean = np.array([np.nanmean(flat_data)])
        full_data_std = np.array([np.nanstd(flat_data)])
        np.savetxt(Full_data_flattened + r"\Full_data_mean.txt", full_data_mean, fmt='%1.2f')
        np.savetxt(Full_data_flattened + r"\Full_data_std.txt", full_data_std, fmt='%1.2f')

