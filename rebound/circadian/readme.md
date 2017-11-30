This repo builds on previous Urban Observatory (UO) work investigating the rebound effect.

It contains code for further image and signal processing of broadband and hyperspectral images collected by the UO in order to investigate the effect of blue light on human circadian rhythms.

**********************
Background literature:
**********************

Lighting for the human circadian clock (Pauly, 2004): Presents hypothesis that light at night (LAN) is a factor in types of cancer due to melatonin suppression.
http://www.sciencedirect.com/science/article/pii/S0306987704002336 

Action spectrum for melatonin regulation in humans (Brainard 2001): experiments using action spectra (a method that compares number of photons at various wavelengths needed to elicit the same given biological effect). Found that light in 446-477 nm wavelength range (blue) most actively suppresses melatonin in humans. Results suggest a single photoreceptor (a.k.a. “cryptochrome” is responsible.)
http://www.jneurosci.org/content/21/16/6405.short

Phototransduction by Retinal Ganglion Cells That Set the Circadian Clock (Berson, 2002): Discovery that retinal ganglion cells (RGC) are responsible for suppressing melatonin in humans, thus identifying a specific physiological mechanism for blue light’s effect on circadian rhythm. Implications are that light intensity in other wavelengths (through conventional conic receptors) are not as influential on circadian rhythm.
http://science.sciencemag.org/content/295/5557/1070

Dynamics of the Urban Landscape: Broadband side-view image analysis of cityscape to identify temporal patterns in lighting.
http://www.sciencedirect.com/science/article/pii/S0306437915001167

A Hyperspectral Survey of NYC Lighting Technology: Hyperspectral image analysis to identify and label sources of light using unsupervised clustering methods and NOAA lab spectra as references.
http://www.mdpi.com/1424-8220/16/12/2047/htm

****
Data
****
Code in this repo processes and analyzes data from two primary sources, as collected by instrumentation at the NYU CUSP Urban Observatory:
1. Broadband images
	- 3-color (RBG) format
	- 10-second time resolution (~3,200 scans/night)
	- 3096 x 4096 pixel resolution
	- collected over ~60 nights

2. Hyperspectral images (HSI)
	- 848-channel format
	- ~ 80-second time resolution (varies)
	- 1600 x ~3200 pixel resolution
	- collected over 20+ nights

*******
Methods
*******
Initial source extraction and integration of broadband and HSI scans via image registration.
- Previous code here: https://github.com/gdobler/rebound/tree/master/rebound

>> precision_stack.py
https://github.com/gdobler/rebound/blob/master/rebound/circadian/precision_stack.py
"Precision stacking" to stack HSI scans only when associated light source is determined to be on, based on the broadband images.

>> cluster_analysis.py
https://github.com/gdobler/rebound/blob/master/rebound/circadian/cluster_analysis.py
Initial unsupervised cluster analysis to identify groups of similar spectra among light sources based on HSI scans and indicate possible classifications of lighting technology, such as High Pressure Sodium, LED, Fluorescent, Metal Hallide.

>> utils.py
https://github.com/gdobler/rebound/blob/master/rebound/circadian/utils.py
Contains various methods to extract and process raw data.
