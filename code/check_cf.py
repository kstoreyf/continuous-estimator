#!/usr/bin/env python
import os
import numpy as np
import time

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic

from nbodykit.lab import *
import nbodykit


Nrealizations = 15

rmin = 40.
rmax = 200.
binwidth = 10.
r_edges = np.arange(rmin, rmax+binwidth, binwidth)
r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

proj_type = 'tophat'
nprojbins = len(r_avg)
projfn = None

for Nr in range(Nrealizations):
    print(f"Realization {Nr}")

    N = 300
    L = 750
    x = np.random.rand(N)*L
    y = np.random.rand(N)*L
    z = np.random.rand(N)*L

    ## WORKS FOR VERBOSE=FALSE
    ## HANGS FOR VERBOSE=TRUE
    ## WHAT THE FUCKKKKK     
    cosmology = 1
    verbose = True
    weight_type = 'pair_product'
    isa = 'fallback'

    pos = np.array([x,y,z]).T
    cosmo = nbodykit.cosmology.Planck15
    ra_data, dec_data, z_data = nbodykit.transform.CartesianToSky(pos, cosmo)
    nmubins = 1
    mumax = 1.0
    nthreads = 24
    dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, r_edges, ra_data, dec_data, z_data, verbose=verbose, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    print(dd_proj)

    # HANGS (?!)
    # mumax = 1.0
    # nmubins = 1
    # verbose = True
    # periodic = True
    # nthreads = 24
    # cosmology = 1
    # cosmo = nbodykit.cosmology.Planck15
    # pos = np.array([x,y,z]).T
    # ra_data, dec_data, z_data = nbodykit.transform.CartesianToSky(pos, cosmo)
    # dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, r_edges, ra_data, dec_data, z_data, verbose=verbose, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)
    # print(dd_proj)

    # hangs
    #_, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, verbose=verbose, boxsize=L, periodic=periodic)
    #print(dd_proj)

    # works
    # res = Corrfunc.theory.xi(L, nthreads, r_edges, x, y, z)
    # res = np.array(res)
    # xi = [rr[3] for rr in res]
    # print(xi)