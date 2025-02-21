# Cluster Lensing 

This package computes the cluster lensing signal and covariance matrices.  Both in terms of $\gamma_t$ and $\Delta\Sigma$.

<!-- Put this in your `~/.bashrc`

`export PYTHONPATH="/your_folder/cluster-lensing-cov/:$PYTHONPATH"` -->

## Installation
First run `git clone https://github.com/hywu/cluster-lensing-cov`

Then run `pip install cluster-lensing-cov/`

## Contents

* `clens/lensing/` contains the main code for computing lensing signals and covariance matrices.  

	* `lensing_profiles.py` calculates $\gamma_t$ and $\Delta\Sigma$ profiles assuming NFW

	* `cov_DeltaSigma.py` and `cov_gammat.py` calculate the covariance matrices  

* `clens/util/` contains classes for cosmology, nuisance parameters, survey conditions, etc.

* `clens/ying/` contains code copied from Ying Zu's package

* `tests/` contains unit tests.

* `examples/` 
	* `demo_analytic.py` shows the one example for 
	* `abacus_analytic_grafting.py` shows how to graft analytic and abacus together 


## Pre-computed covariance matrices

* `output/analytic_abacus_scatter*/zh*_zs*/*_*_R*_*_nrp*/` includes various components of covariance matrices.
	* `DeltaSigma_cov_combined.dat` is everything without shape noise
	* `DeltaSigma_cov_shape_noise.dat` is for nsrc = 10 arcmin^{-2}
	*  Shape noise is inversely proportional to nsrc

* For example, `output/analytic_abacus_scatter0/zh0.5_zs1.25/1e+14_1e+16_R0.1_100_nrp15/` contains zero scatter, zh=0.5, zs=1.25, M between 1e14 and 1e16 Msun/h, 15 log rp bins between 0.1 and 100 Mpc/h