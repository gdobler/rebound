# rebound
Empirical Measurement of the Rebound Effect

/bb --> directory for processing broadband
- srcex.py: source extraction using temporal correlation
- get_lightcurves.py: extract lightcurves using source mask produced by srcex.py
- detect_onoff.py: edge detection method for identifying on / off transitions in lightcurves produced by get_lightcuves.py
- duration.py: calculate duaration of lights on time based on on / off tags produced in detect_onoff.py
- plotting.py: varous methods to plot broadband images and data
- bb_settings.py: global variables, pathfiles, etc (requires setting local environmental variables)
- utils.py: various processing methods
- various scripts to register broadband and HSI images to produce final mask

/hsi --> directory for processing hyperspectral images
- various scripts to stack and classify light types using HSI scans

/ext --> directory for 3-d model projection
- various scrpts to project 2-d light source coordinates into 3-d data model of built environment

/utils --> directory for various processing methods, such as postgres db, hdf5 compression, etc

