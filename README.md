# Nanosensitive Optical Coherence Tomography Python Scripts

This repo contains all the necessary python scripts to peform the original nanosesitive OCT (nsOCT) algorithm. This reclaims high axial spatial frequency information that is 
lost after Fourier transformation of the spectral densitiy of the OCT signal. This is achieved by spectrally encoding the spatial frequency information through spectral diversity, translate
it from the Fourier domain to image domain, and map each voxel of the 3D image. In this case, the dominant spatial frequency of each voxel is taken.

### Repo Contents:

- Process OCT signal and performing nsOCT processing on the spectral information found in raw data
- Plot OCT and nsOCT data to generate B-framess and en-face image
- Find the mean and standard deviation of mean values for each B-frame and en-face images
- Plot histograms, depth plots, and flattended plot of entire data
- Dictionary of parameters and main program scripts
- A windowing script that contains various different window functions for OCT processing

### How to use:
- Edit Params.py script to match specifications of OCT system being used and imaging sample
- Load in file(s) through f_name found in Params.py
- Select which scripts to run in Main_Program.py

