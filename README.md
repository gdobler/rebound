# rebound
Empirical Measurement of the Rebound Effect

***********
Quick reference for scripts in /rebound
**********

plotting.py

Requires 3 input arguments: datacube, raw image file, and a parameter known as "wave_bin" which is an integer from 0 to 871 indicating the particular spectral channel that will be displayed in the lower plot.

util scripts produce an output that is a class object, e.g. saved as "cube". This variable should be used as the first input in plotting.hyper_viz(). Calling the .data method on it, e.g. "cube.data" will produce the 2nd input (raw image data). Third input is entered by programmer.

hdf5_compress.py

Compresses raw image data into hdf5 format (a work in progress, currently only converts broadband from the rebound test images taken in May, edits welcome). Requires 3 inputs: 
- full filepath for the raw image data, 
- shape (which is just a tuple of height x width for May test images), and
- full filepath to output folder for hdf5 file object (this folder requires 2 sub-folder be created before running script: /hyperspectral and /broadband
You can also set arguments for compression ratio, etc. It calls the .read_raw() method from util_test automatically and outputs an hdf5 file object.

    dependencies are:
    - util_test 
    - the Python library h5py

