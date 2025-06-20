from Params import*
import numpy as np
import nsOCT_maximum_spatial_period as Basic
import plot_B_frames as Bframe
import plot_enface_images as enface
import B_frame_mean_spatial_period as B_mean
import enface_mean_spatial_period as en_mean
import ful_data_flattened as flat

num_of_files = len(Params["f_name"]) # Reads in multiple files
for n in np.arange(0, num_of_files):
    print('processing data {}'.format(n + 1), 'out of {}'.format(num_of_files))
    f_name = Params["f_name"][n]
    print(f_name)
    three_d_OCT, three_d_nsOCT,bins = Basic.oct_ns_oct_processing(f_name)
    #Bframe.plot_b_frames(oct=three_d_OCT, nsoct=three_d_nsOCT, do_plot=True,do_save=True)
    #enface.plot_enface_images(oct = three_d_OCT, nsoct = three_d_nsOCT, do_plot = True, do_save = True)
    B_mean.mean_spatial_period_for_b_frame(nsoct = three_d_nsOCT, bins = bins, do_compute = True, do_save = True)
    en_mean.mean_spatial_period_for_enface(nsoct = three_d_nsOCT,bins = bins, do_compute = True, do_save = True)
    flat.full_data_flattened(nsoct=three_d_nsOCT, bins = bins, do_compute=True)
