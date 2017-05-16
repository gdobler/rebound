# rebound
Empirical Measurement of the Rebound Effect

***********
Quick reference for scripts in /rebound
**********

plotting.py

Requires 3 input arguments: datacube, raw image file, and a parameter known as "wave_bin" which is an integer from 0 to 871 indicating the particular spectral channel that will be displayed in the lower plot.

util scripts produce an output that is a class object, e.g. saved as "cube". This variable can be input as the first input in plotting.hyper_viz. Calling the .data method on it, e.g. "cube.data" will produce the 2nd input (raw image data). Third input is entered by programmer.
