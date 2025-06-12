f_name =[r'/Users/tomi/Desktop/3D_nsOCT_code/oct_data/7D_DAY1_0005_Mode3D.oct',
         r'/Users/tomi/Desktop/3D_nsOCT_code/oct_data/3D_healthy_0004_Mode3D.oct']
Number_of_windows = 12
Threshold = 2.2
oversampling_degree = 1
B_frames_to_process = [1,500]
A_lines_to_process = [1,500]
Lower_wavelength = 1.180
Upper_wavelength = 1.415
Refractive_index = 1.00
pixels_selected= [1,300]
histograms_y_axis_limit = 0.6
histograms_x_axis_range = [590,700]
nsOCT_image_colorbar = [600,680]
OCT_image_intensity_range = [140,240]
B_frame_aspect_ratio = 1
en_face_aspect_ratio = 1

Params ={'f_name': f_name,
         'Threshold':Threshold,
         'Number_of_windows': Number_of_windows,
         'Refractive_index' : Refractive_index,
         'B_frames_to_process' : B_frames_to_process,
         'Lower_wavelength' : Lower_wavelength,
         'Upper_wavelength' : Upper_wavelength,
         'pixels_selected' : pixels_selected,
         'A_lines_to_process':A_lines_to_process,
         'histograms_x_axis_range': histograms_x_axis_range,
         'histograms_y_axis_limit':histograms_y_axis_limit,
         'nsOCT_image_colorbar': nsOCT_image_colorbar,
         'OCT_image_intensity_range': OCT_image_intensity_range,
         'B_frame_aspect_ratio': B_frame_aspect_ratio,
         'en_face_aspect_ratio': en_face_aspect_ratio}

