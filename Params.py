f_name = ["/Users/tomi/Desktop/3D_nsOCT_code/oct_data/3D_healthy_0004_Mode3D.oct"]
number_of_windows = 12  # zones for nsOCT
threshold = 2.2
oversampling_degree = 1  # read paper by Zhou et al. 2019
b_frames_to_process = [1,10]
a_lines_to_process = [1,500]
lower_wavelength = 1.180
upper_wavelength = 1.415
refractive_index = 1.00
pixels_selected = [50,52]
histograms_y_axis_limit = 0.6
histograms_x_axis_range = [590,700]
ns_oct_image_colorbar = [600,680]
oct_image_intensity_range = [140,240]
b_frame_aspect_ratio = 1
en_face_aspect_ratio = 1

Params = {
    'f_name': f_name,
    'threshold': threshold,
    'number_of_windows': number_of_windows,
    'refractive_index': refractive_index,
    'b_frames_to_process': b_frames_to_process,
    'lower_wavelength': lower_wavelength,
    'upper_wavelength': upper_wavelength,
    'pixels_selected': pixels_selected,
    'a_lines_to_process': a_lines_to_process,
    'histograms_x_axis_range': histograms_x_axis_range,
    'histograms_y_axis_limit': histograms_y_axis_limit,
    'ns_oct_image_colorbar': ns_oct_image_colorbar,
    'oct_image_intensity_range': oct_image_intensity_range,
    'b_frame_aspect_ratio': b_frame_aspect_ratio,
    'en_face_aspect_ratio': en_face_aspect_ratio,
    'oversampling_degree': oversampling_degree
}