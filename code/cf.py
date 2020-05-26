#!/usr/bin/env python
import os
import numpy as np
import time

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import spline
from Corrfunc.bases import bao

import read_lognormal as reader

from Corrfunc.theory.DD import DD
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from nbodykit.lab import *
import nbodykit

def main():
    cat_tag = '_L750_n1e-4'

    #proj = 'theory'
    #kwargs = {'binwidth': 2.5}
    #cf_tag = f"_{proj}_bw{kwargs['binwidth']}"

    proj = 'tophat'
    kwargs = {'binwidth': 10}
    cf_tag = f"_{proj}_bw{kwargs['binwidth']}"

    # proj = 'spline'
    # binwidth = 10
    # kwargs = {'order': 1}
    # cf_tag = f"_{proj}{kwargs['order']}_bw{binwidth}"

    nbins = None
    Nrealizations = 20
    overwrite = True

    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # define bins
    binwidth = kwargs['binwidth']
    rmin = 40.
    rmax = 200.
    assert bool(nbins) ^ bool(binwidth), "Can't set both nbins and binwidth!"
    if nbins:   
        binwidth = (rmax-rmin)/float(nbins) 
    r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

    proj_type, nprojbins, projfn = get_proj_parameters(proj, r_edges=r_edges, **kwargs)

    cat_dir = '../catalogs/lognormal'
    for Nr in range(Nrealizations):
        print(f"Realization {Nr}")
        save_fn = f'{result_dir}/cf{cf_tag}{cat_tag}_rlz{Nr}.npy'
        if os.path.exists(save_fn) and not overwrite:
            print(f"Already exists! ({save_fn}) continuing")
            continue
    
        print("Computing xi")
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{Nr}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        x, y, z, vx, vy, vz = data.T
        # N = 300
        # Lx = 750
        # x = np.random.rand(N)*Lx
        # y = np.random.rand(N)*Lx
        # z = np.random.rand(N)*Lx
        print(N)
        
        start = time.time()
        if "theory" in proj:
        #if True:
            xi = xi_theory_periodic(x, y, z, Lx, r_edges)
            r = r_avg
        else:
            r, xi = xi_proj_periodic(x, y, z, Lx, r_edges, nprojbins, proj_type, projfn=projfn)
        end = time.time()

        np.save(save_fn, [r, xi, proj])
        print("Time:", end-start, "s")
            

def xi_theory_periodic(x, y, z, L, r_edges):
    nthreads = 24
    res = Corrfunc.theory.xi(L, nthreads, r_edges, x, y, z)
    res = np.array(res)
    xi = [rr[3] for rr in res]
    print("THEORY")
    print([rr[4] for rr in res])
    print(xi)
    return xi


def xi_proj_periodic(x, y, z, L, r_edges, nprojbins, proj_type, projfn=None):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    periodic = True
    nthreads = 24
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                verbose=verbose, boxsize=L, periodic=periodic)
    

    # cosmology = 1
    # cosmo = nbodykit.cosmology.Planck15
    # pos = np.array([x,y,z]).T
    # ra_data, dec_data, z_data = nbodykit.transform.CartesianToSky(pos, cosmo)

    # dd_res_corrfunc, dd_proj, dd_projt = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubins, rbins, ra_data, dec_data, z_data, verbose=verbose, proj_type=proj_type, nprojbins=nprojbins, projfn=projfn)

    #dd = DD(1, nthreads, rbins, X1=x, Y1=y, Z1=z,
    #           periodic=periodic, boxsize=L)
    #dd_proj = np.array([x[3] for x in dd])

    rmin = min(r_edges)
    rmax = max(r_edges)
    volume = L**3
    nrbins = len(r_edges)-1
    nd = len(x)
    rr_ana, qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nrbins, r_edges, proj_type, projfn=projfn)

    rcont = np.linspace(rmin, rmax, 1000)
    #r_avg = 0.5*(r_edges[1:]+r_edges[:-1])
    #rcont = r_avg
    amps_periodic_ana = np.linalg.solve(qq_ana, dd_proj) - 1
    #amps_periodic_ana = np.matmul(np.linalg.inv(qq_ana), dd_proj) - 1
    xi_periodic_ana = evaluate_xi(nrbins, amps_periodic_ana, len(rcont), rcont, len(r_edges)-1, r_edges, proj_type, projfn=projfn)
    
    amps_periodic_ana_inv = np.matmul(np.linalg.inv(qq_ana), dd_proj) - 1
    xi_periodic_ana_inv = evaluate_xi(nrbins, amps_periodic_ana_inv, len(rcont), rcont, len(r_edges)-1, r_edges, proj_type, projfn=projfn)


    print("PROJ")
    print(dd_proj)
    # print(rr_ana)
    # print(dd_proj/rr_ana - 1)
    # print(xi_periodic_ana)
    # print(xi_periodic_ana_inv)
    return rcont, xi_periodic_ana
    #return 1, 2
    #return rcont, dd_proj

def get_proj_parameters(proj, r_edges=None, **kwargs):
    proj_type = proj
    projfn = None
    if proj=="tophat" or proj_type=="piecewise":
        nprojbins = len(r_edges)-1
    elif proj=='spline':
        nprojbins = len(r_edges)-1
        proj_type = 'generalr'
        projfn = '../tables/spline.dat'
        spline.write_bases(r_edges[0], r_edges[-1], len(r_edges)-1, projfn, **kwargs)
    elif proj=='bao':
        proj_type = 'generalr'
        projfn = '../tables/bao.dat'
        print(kwargs)
        nprojbins, _ = bao.write_bases(r_edges[0], r_edges[-1], projfn, **kwargs) 
    elif proj=='theory':
        proj_type = None
        nprojbins = None
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    return proj_type, nprojbins, projfn


if __name__=='__main__':
    main()
