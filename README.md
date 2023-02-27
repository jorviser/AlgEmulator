# AlgEmulator
Atmospheric Look-up table Generator (ALG) emulator tool.

This repository contains a set of scripts to help construct emulators for atmospheric radiative transfer models (RTMs). In order to be compliant with the required algEmulator interfaces, the RTM-based training datasets should be generated with the ALG toolbox:

J. Vicent, J. Verrelst, N. Sabater, L. Alonso, J.P. Rivera-Caicedo, L. Martino, J. Muñoz-Marí and J. Moreno: "Comparative analysis of atmospheric radiative transfer models using the Atmospheric Look-up table Generator (ALG) toolbox (version 2.0)", Geosci. Model Dev., 13, 1945–1957, https://doi.org/10.5194/gmd-13-1945-2020, 2020. 

The original algEmulator work was presented in:

J. Vicent et al., "Systematic Assessment of MODTRAN Emulators for Atmospheric Correction," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 4101917, https://doi.org/10.1109/TGRS.2021.3071376.

and further expanded with the multi-fidelity emulation options:

J. Vicent et al., "Multi-fidelity Gaussian Process Emulation for Atmospheric Radiative Transfer Models," in IEEE Transactions on Geoscience and Remote Sensing, vol. X, pp. X-XX, 2023, Art no. XXXXXX,.

We appreciate you citing the above publications when using the algEmulator tool.

Note that algEmulator is currently only working in Matlab. Future evolution for python implementation is on-going.

## Getting started

Start by cloning this git repository in your local computer:
```
git clone https://gitlab.com/jorge.vicent/algEmulator.git
```

After cloning, you will find the following files:
- This README.md file.
- algEmulator.m: a Matlab object/function for training emulators and prediction.
- analysis_script.m: a Matlab script to show how to use algEmulator in the practical example of atmospheric correction of PRISMA L1 data.

A trained algEmulator for MODTRAN6 can be found at DOI: 10.5281/zenodo.XXXXX together with the datasets for training and testing.

## Usage
The algEmulator object can be initialized as follows:
```
emu = algEmulator(file,conf);
```
where ```file``` is the filepath/name of a look-up table file (.h5) generated with ALG and ```conf``` is a Matlab structure with the configuration of the emulator, and it contains the following fields:
- method: method to run the emulators: 'alg' (default) or 'matlab' (recommended only for validation).
- kernel: kernel function for GPR: 'squaredexponential' (default) or 'ardsquaredexponential'.
dimReduction: method for dimensionality reduction: 'pca' (default) for PCA decomposition or 'none'. Note that the 'none' option can be computationally expensive since one GPR is trained for each output wavelength.
- numCmp: number of components after dimensionality reduction. If dimReduction='pca', this value can be defined as a scalar (=20, default value) or a vector defining the number of PCA components for each atmospheric transfer function. If dimReduction='none', this value is automatically replaced by the number of wavelengths unless multifidelify.method='linear', in which case numCmp=2 by definition.
- multifidelity: Structure defining the configuration of a multifidelity emulator. The following parameters can be configured: (1) **LFregressor**: low-fidelity (LF) regression model among the options 'polyfit' (default, Ndim surface fitting by 2nd order polynomial) or 'gpr' (with a squaredexponential kernel and 2 PCAs). (2) **method**: 'delta' (default): a GP emulator is trained for the difference between the real (training) and the LF regression data. Note that this difference is spectral (multioutput)! 'gain': a GP emulator is trained for ratio between the real (training) and the LF  regression data. Note that this ratio is spectral (multioutput)! 'linear': a linear regression between the LF regression and the real (training) data is calculated (Yhf = A.Ylf+B). The GP emulator is trained for the values of the fitting coefficients A and B. Note that this coefficients are scalars!. (3) **nLayers**: integer value (>0) defining the number of multi-fidelity layers. This allows to execute the lowest fidelity with LFregressor and nLayers-1 recursive GPs with the same configuration as in conf. If multifidelity is not used if is this parameter is empty, non-defined or any other non-valid value.

Once the emulator is trained, it can be used to predict atmospheric transfer functions (i.e. transmittances, spherical albedo and path radiance) as follows:
```
Yq = emu.emulate(Xq);
```
where ```Xq``` is a matrix with N query points each of them of dimension D (i.e. the number of atmospheric/geometric variables in the input space). The output ```Yq``` is a matrix of dimensions NxLxP where L is the number of wavelengths and P is the number of transfer functions (typically 6). If you show ```emu``` in Matlab's command window, you'll see the complete list of attributes, including outnames (for interpreting the output spectra) and varnames (names of input variables)

## Support
If you find any issues in the algEmulator or you have questions, please don't hesitate to contact us at jorge.vicent@uv.es 

## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

## Project status
Future evolutions of algEmulator might be released in the future. Stay tuned!