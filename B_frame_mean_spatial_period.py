import numpy as np
import matplotlib.pyplot as plt
from Params import*
from nsOCT_maximum_spatial_periodOCT import *



def Mean_spatial_period_for_B_frame(nsOCT, bins, do_compute, do_save):

    if do_compute:

        pixels_selected = Params['pixels_selected']
        Number_of_A_lines = Params['Number_of_A_lines']
        Number_of_B_frames = Params['Number_of_B_frames']
        Lateral_dimension_X = Params['Lateral_dimension_X']
        Lateral_dimension_Y = Params['Lateral_dimension_Y']
        Depth_in_mm = Params['Depth_in_mm']
        B_frame_images = Params['B_frame_images']
        threshold_spatial_period = Params['threshold_spatial_period']
        B_frame_histograms = Params['B_frame_histograms']
        B_frames_to_process = Params["B_frames_to_process"]

        three_d_nsOCT = nsOCT


        processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1]+1)


        """
        For plotting Histogram
        """


        NaN_limit = threshold_spatial_period

        three_d_nsOCT[three_d_nsOCT <= NaN_limit] = np.NaN

        three_d_nsOCT=three_d_nsOCT[:, pixels_selected[0]-1: pixels_selected[1], :]




        Normal_B_frame_averaged_spatial_period = np.nanmean(three_d_nsOCT, axis=0)
        print("The dimension of Mean_nsOCT_image is", np.shape(Normal_B_frame_averaged_spatial_period))
        flat_B_frame = Normal_B_frame_averaged_spatial_period.flatten()
        Normal_Mean_spatial_period_of_data_from_averaged_B_frame = np.nanmean(flat_B_frame)
        Normal_std_of_spatial_period_data = np.nanstd(flat_B_frame)

        plt.hist(flat_B_frame, bins=bins, range=(histograms_x_axis_range[0], histograms_x_axis_range[1]),density=True,  histtype='bar', color='c',rwidth=0.8)
        plt.tight_layout()
        plt.title("Mean of all B-frames")
        plt.xlabel("Spatial period (nm)")
        plt.ylabel("Amplitude (a.u)")
        plt.ylim(0, histograms_y_axis_limit)
        plt.tight_layout()
        script_dir = os.path.dirname(__file__)
        b_frame_hist_dir = os.path.join(script_dir, B_frame_histograms)
        plt.savefig(os.path.join(b_frame_hist_dir, "{}Averaged_B_frame.png"))
        plt.clf()

        Single_mean_and_std_from_averaged_B_frame = np.array([Normal_Mean_spatial_period_of_data_from_averaged_B_frame, Normal_std_of_spatial_period_data])
        print("Normal single mean spatial period of data and spatial period are ", Normal_Mean_spatial_period_of_data_from_averaged_B_frame,
              Normal_std_of_spatial_period_data)

        np.savetxt(B_frame_histograms + r"\B_frame_single_data_mean_and_std.txt", Single_mean_and_std_from_averaged_B_frame, fmt='%1.2f')

        """Average of a single B_frame"""

        B_frame_means = []
        B_frames_std = []
        for B_frame,histo_name in zip(three_d_nsOCT,processing_B_frames):
            B_frame_averaged_along_A_lines = np.nanmean(B_frame, axis = 1)
            flattened_B_frame = B_frame.flatten()
            B_frame_mean = np.nanmean(flattened_B_frame)
            B_frame_means.append(B_frame_mean)
            B_frame_std = np.nanstd(flattened_B_frame)
            B_frames_std.append(B_frame_std)

            if do_save:
                B_frame_histo = plt.hist(B_frame_averaged_along_A_lines,bins = bins,rwidth= 0.8,density=True, range=(histograms_x_axis_range[0], histograms_x_axis_range[1]), histtype='bar', color='g')
                # sns.distplot(flattened_B_frame)
                plt.title("Spatial period distribution for a B-frame")
                plt.xlabel('Spatial period (nm)')
                plt.ylabel('Amplitude (a.u)')
                plt.ylim(0,histograms_y_axis_limit)
                plt.tight_layout()
                script_dir = os.path.dirname(__file__)
                b_frame_hist_dir = os.path.join(script_dir, B_frame_histograms)
                plt.savefig(os.path.join(b_frame_hist_dir, "{}.png".format(histo_name)))
                plt.clf()

        B_frame_means = np.asarray(B_frame_means)
        B_frames_std = np.asarray(B_frames_std)
        mean_from_B_frame_means = np.mean(B_frame_means)
        mean_from_B_frame_means = np.asarray([mean_from_B_frame_means])
        standard_deviation_from_B_frame_means = np.std(B_frame_means)
        standard_deviation_from_B_frame_means = np.asarray([standard_deviation_from_B_frame_means])
        mean_vs_B_frames = plt.plot(processing_B_frames,B_frame_means, '.')
        plt.title("b-frames vs mean spatial period")
        plt.xlabel('B-frame number')
        plt.ylabel("Mean spatial period for B-frame(nm)")
        plt.tight_layout()

        script_dir = os.path.dirname(__file__)
        b_frame_hist_dir = os.path.join(script_dir, B_frame_histograms)
        plt.savefig(os.path.join(b_frame_hist_dir, "mean_vs_B_frames.png"))
        plt.clf()
        np.save(B_frame_histograms + r"\B_frame_means.npy", B_frame_means)
        np.savetxt(B_frame_histograms + r"\std_from_B_frame_means.txt",standard_deviation_from_B_frame_means, fmt='%1.2f')
        np.savetxt(B_frame_histograms + r"\mean_from_B_frame_means.txt", mean_from_B_frame_means, fmt='%1.2f')
        np.savetxt(B_frame_histograms + r"\B_frame_stds.txt", B_frames_std, fmt='%1.2f')
        np.savetxt(B_frame_histograms + r"\B_frame_means.txt" , B_frame_means,fmt='%1.2f')
        np.savetxt(B_frame_histograms + r"\B_frame_stds.txt" , B_frames_std, fmt='%1.2f')


