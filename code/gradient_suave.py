#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from nbodykit.lab import *
import nbodykit
import suave
from suave.theory.DD import DD
from suave.theory.DDsmu import DDsmu
from suave.mocks.DDsmu_mocks import DDsmu_mocks
from suave.utils import evaluate_xi
from suave.utils import qq_analytic
from suave.bases import spline
from suave.bases import bao
from suave.utils import compute_amps

import read_lognormal as reader


def main():
    print("Go!")

    print(suave.__version__)
    print(suave.__file__)
    nrealizations = 1
    L = 750
    #L = 450.0
    N = 1000 # number of points in mock
    cat_tag = 'cattest'  
 
    proj = 'gradient'
    kwargs = {}
    binwidth = 10
    cf_tag = f"_{proj}_top_bw{binwidth}"

    #proj = 'tophat'
    #kwargs = {}
    #binwidth = 5
    #cf_tag = f"_{proj}_bw{binwidth}_anatest"

    #proj = None
    #kwargs = {}
    #binwidth = 10
    #cf_tag = f"_{proj}_bw{binwidth}"

    rmin = 40.0
    rmax = 60.0
    r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    rmax = max(r_edges)
    nrbins = len(r_edges)-1
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)    

    proj_type, nprojbins, projfn = get_proj_parameters(proj, r_edges=r_edges, cf_tag=cf_tag, **kwargs)

    nthreads = 2

    for Nr in range(nrealizations):

        print(f"Realization {Nr}")
        save_fn = f'{result_dir}/cf{cf_tag}_rlz{Nr}.npy'

        # Generate cubic mock
        x = np.random.rand(N)*float(L)
        y = np.random.rand(N)*float(L)
        z = np.random.rand(N)*float(L)
        ones = np.ones(N)
        if proj_type=='gradient':
            weights = np.array([ones, x, y, z])        
        else:
            weights = ones

        # Generate random catalog
        randmult = 2
        Nr = N*randmult
        xr = np.random.rand(Nr)*float(L)
        yr = np.random.rand(Nr)*float(L)
        zr = np.random.rand(Nr)*float(L)
        ones = np.ones(Nr)
        if proj_type=='gradient':
            weights_r = np.array([ones, xr, yr, zr])
        else:
            weights_r = ones

        print("Run DDsmu, {} basis".format(proj))
        nmubins = 1
        mumax = 1.0
        periodic = True
        verbose = True
        weight_type = 'pair_product_gradient'
        np.set_printoptions(precision=6)

        dd_res, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, periodic=periodic, boxsize=L, verbose=verbose, weight_type=weight_type)
        print("DD:", np.array(dd_proj))

        dr_res, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, weights1=weights, X2=xr, Y2=yr, Z2=zr, weights2=weights_r, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, periodic=periodic, boxsize=L, verbose=verbose, weight_type=weight_type)
        print("DR:", np.array(dr_proj))

        rr_res, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, xr, yr, zr, weights1=weights_r, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn, periodic=periodic, boxsize=L, verbose=verbose, weight_type=weight_type)
        print("RR:", np.array(rr_proj))

        if proj_type is None:
            print(dd_res)
            continue

        amps = compute_amps(nprojbins, N, N, Nr, Nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
        print(amps)
        
        #weights1 = np.array([1.0, 325., 325., 325.])
        #weights2 = np.array([1.0, 325., 325., 325.])
        weights1 = np.array([1.0, 750., 750., 750.])
        weights2 = np.array([1.0, 750., 750., 750.])
        rcont = np.linspace(rmin, rmax, 10)

        #weights1=None
        #weights2=None
        #weight_type=None
        
        print(amps, rcont, proj_type, r_edges, projfn, weights1, weights2, weight_type)
        xi = evaluate_xi(amps, rcont, proj_type, rbins=r_edges, projfn=projfn, weights1=weights1, weights2=weights2, weight_type=weight_type)
        # for now! 
        print(xi)
        r = []
        xi = []
        extra_dict = {}
        np.save(save_fn, [r, xi, amps, proj, extra_dict])

        #print("Run qq_analytic")
        #rmin = min(r_edges)
        #rmax = max(r_edges)
        #nd = len(x)
        #volume = float(L**3)
        #nrbins = len(r_edges)-1 
        #rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nrbins, r_edges, proj_type, projfn=projfn)
        #print("RR_ana:", rr_ana)

        #print("Run evaluate_xi")
        #rcont = np.linspace(rmin, rmax, 1000)
        #numerator = dd_proj - rr_ana
        #amps_periodic_ana = np.linalg.solve(qq_ana, numerator)
        #print("amplitudes:", amps_periodic_ana)
        #xi_periodic_ana = evaluate_xi(nprojbins, amps_periodic_ana, len(rcont), rcont, nrbins, r_edges, proj_type, projfn=projfn)
        #print("evaluate_xi done")





def get_proj_parameters(proj, r_edges=None, cf_tag=None, **kwargs):
    proj_type = proj
    projfn = None
    if proj=="tophat" or proj_type=="piecewise":
        nprojbins = len(r_edges)-1
    elif proj=='spline':
        nprojbins = len(r_edges)-1
        proj_type = 'generalr'
        projfn = f'../tables/bases{cf_tag}.dat'
        spline.write_bases(r_edges[0], r_edges[-1], len(r_edges)-1, projfn, **kwargs)
    elif proj=='bao':
        proj_type = 'generalr'
        projfn = f'../tables/bases{cf_tag}.dat'
        print(kwargs)
        nprojbins, _ = bao.write_bases(r_edges[0], r_edges[-1], projfn, **kwargs) 
    elif proj=='gradient':
        nprojbins = 4*(len(r_edges)-1)
    elif proj==None:
        nprojbins = None
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    return proj_type, nprojbins, projfn


if __name__=='__main__':
    main()
