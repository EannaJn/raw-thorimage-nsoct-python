# Nanosensitive Optical Coherence Tomography Python Scripts

This repo contains all the necessary python scripts to peform the original nanosesitive OCT (nsOCT) algorithm. This reclaims high axial spatial frequency information that is 
lost after Fourier transformation of the spectral densitiy of the OCT signal. This is achieved by spectrally encoding the spatial frequency information through spectral diversity, translate
it from the Fourier domain to image domain, and map each voxel of the 2D image. In this case, the dominant spatial frequency of each voxel is taken.

### Repo Contents:

- Processing on OCT signal and performing nsOCT processing on the spectral information found in raw data
- Plotting of OCT signal and nsOCT data to generate B-framess and en-face image
- Find the mean and standard deviation of mean values for each B-frame and en-face images
- Plot histograms, depth plots, and flattended plot of entire data
- Dictionary of parameters and main program scripts
- A windowing script that contains various different window functions for OCT processing
