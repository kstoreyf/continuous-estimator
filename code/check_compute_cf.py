#!/usr/bin/env python
import numpy as np
import os

from nbodykit.lab import *
import nbodykit

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks

import corrfuncproj


Nrealizations = 15

rmin = 40.
rmax = 200.
binwidth = 10.
r_edges = np.arange(rmin, rmax+binwidth, binwidth)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

proj_type = 'tophat'
nprojbins = len(r_avg)
projfn = None

def to_sky(pos, cosmo, velocity=None, rsd=False, comoving=True):
    if rsd:
        if velocity is None:
            raise ValueError("Must provide velocities for RSD! Or set rsd=False.")
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    else:
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo)
    if comoving:
        z = cosmo.comoving_distance(z)
    return np.array(ra), np.array(dec), np.array(z)


for Nr in range(Nrealizations):
    print(f"Realization {Nr}")

    N = 300
    L = 750
    x = np.random.rand(N)*L
    y = np.random.rand(N)*L
    z = np.random.rand(N)*L

    #datasky = np.array([ra, dec, cz]).
    cosmology = 1
    verbose = False
    weight_type = 'pair_product'
    isa = 'fallback'

    # FUCKING WORKS
    pos = np.array([x, y, z]).T
    cosmo = nbodykit.cosmology.Planck15
    ra_data, dec_data, z_data = nbodykit.transform.CartesianToSky(pos, cosmo)
    nmubins = 1
    mumax = 1.0
    nthreads = 24
    dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, r_edges, ra_data, dec_data, z_data, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    print(dd_proj)

    # # WORKS
    # pos = np.array([x, y, z]).T
    # cosmo = nbodykit.cosmology.Planck15
    # ra, dec, z = to_sky(pos, cosmo)
    # weights = np.ones(len(ra))
    # nmubins = 1
    # mumax = 1.0
    # nthreads = 24
    # comoving = True
    # dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, r_edges, ra, dec, z,
    # is_comoving_dist=comoving, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    # print(dd_proj)

    # WORKS
    # weights = np.ones(len(ra))
    # nmubins = 1
    # mumax = 1.0
    # nthreads = 24
    # comoving = True
    # dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, r_edges, ra, dec, z,
    #                                weights1=weights, is_comoving_dist=comoving, verbose=verbose,
    #                                weight_type=weight_type, isa=isa, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    # print(dd_proj)


    # WORKS
    # weights = None
    # qq = False
    # cosmology = 1
    # res = corrfuncproj.counts_smu_auto(ra, dec, cz, 
    #                               r_edges, mumax, cosmology, nproc=nthreads,
    #                               weights=weights,
    #                               comoving=True, proj_type=proj_type,
    #                               nprojbins=nprojbins, projfn=projfn, qq=qq)

    # dd_proj, dd_res_corrfunc = res
    # print(dd_proj)

    # hangs
    # nmubins = 1
    # verbose = True
    # periodic = True
    # _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, verbose=verbose, boxsize=L, periodic=periodic)
    # print(dd_proj)

   