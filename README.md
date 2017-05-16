# rebound
Empirical Measurement of the Rebound Effect

***********
Quick reference for scripts in /rebound
**********

util_test.py

Reads raw data from rebound test images. 2 key methods:
- For broadband image data: call util_test.read_raw() with 2 required inputs: the pathfile for the raw image and a tuple of array dimensions.
FOR MAY TEST IMAGES, the tuple parameters in pixels should be: (3072,4096)

- For hyperspectral image data: call util_test.read_hyper() with 1 required input, the full pathfile for the raw hyperspectral image file. The method will automatically call .read_raw() with needed inputs and will produce an output of class "cube", which can be used as inputs for plotting.py as noted below.



plotting.py

Requires 3 input arguments: datacube, raw image file, and a parameter known as "wave_bin" which is an integer from 0 to 871 indicating the particular spectral channel that will be displayed in the lower plot.

util scripts produce an output that is a class object, e.g. saved as "cube". This variable should be used as the first input in plotting.hyper_viz(). Calling the .data method on it, e.g. "cube.data" will produce the 2nd input (raw image data). Third input is entered by programmer.

bband_plot.py

If you just want to view a broadband image, call bband_plot.plot_image() which takes just 1 argument, the variable output from .read_raw().
