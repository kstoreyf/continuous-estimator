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
import bao_utils

from Corrfunc.theory.DD import DD
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from nbodykit.lab import *
import nbodykit




def main():

    print("Corrfunc version:", Corrfunc.__version__, Corrfunc.__file__)

    L = 750
    #nbar_str = '5e-5'
    nbar_str = '2e-4'
    Nrealizations = 100
    cat_tag = f'_L{L}_n{nbar_str}_z057_patchy'
    periodic = False
    overwrite = True
    nx = 3 #for random catalog
    kwargs = {}
    
    proj = 'gradient'
    binwidth = 10 #for bao, still need as dummy parameter
    #cf_tag = f"_{proj}_top_bw{binwidth}_nonperiodic"
    kwargs.update({'cf_tag_bao':'_baoiter_cosmob17_adaptive2'})
    cf_tag = f"_{proj}_bao_rand{nx}x"

    compute_xis(L, nbar_str, cat_tag, proj, cf_tag, binwidth=binwidth, kwargs=kwargs, Nrealizations=Nrealizations, qq_analytic=False, periodic=periodic, nx=nx, overwrite=overwrite)


def compute_xis(L, nbar_str, cat_tag, proj, cf_tag, binwidth=None, nbins=None, kwargs=None, Nrealizations=1000, overwrite=False, qq_analytic=True, nthreads=24, rmin=36.0, rmax=156.0, periodic=False, nx=3):

    proj_type = 'gradient'
    loc_pivot = [L/2., L/2., L/2.]
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    rmin = rmin
    rmax = rmax
    assert bool(nbins) ^ bool(binwidth), "Set either nbins or binwidth (but not both)!"
    if nbins:   
        binwidth = (rmax-rmin)/float(nbins) 
    r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    r_avg = 0.5*(r_edges[1:]+r_edges[:-1])

    # Load in randoms
    rand_dir = '../catalogs/randoms'
    rand_fn = '{}/rand_L{}_n{}_{}x.dat'.format(rand_dir, L, nbar_str, nx)
    random = np.loadtxt(rand_fn)
    rx, ry, rz = random.T
    weights_r = np.array([np.ones(len(rx)), rx-loc_pivot[0], ry-loc_pivot[1], rz-loc_pivot[2]])
    #weights_r = np.array([np.ones(len(rx)), rx, ry, rz])

    cat_dir = '../catalogs/lognormal'
    for Nr in range(Nrealizations):

        extra_dict = {}
        extra_dict.update({'loc_pivot': loc_pivot})

        print(f"Realization {Nr}")
        if 'top' in cf_tag:
            projfn = None
            nprojbins = 4*(len(r_edges)-1)
            rlz_tag = ''
        elif 'bao' in cf_tag:
            kwargs['Nr'] = Nr
            projfn, nprojbins, projfn_bao, amps_bao = bao_utils.get_gradient_bao_params(cat_tag, cf_tag, **kwargs)
            rlz_tag = f'_rlz{Nr}'
            extra_dict.update({'projfn_bao': projfn_bao, 'amps_bao': amps_bao})
            
        weight_type = 'pair_product_gradient'

        # compute random-random vector
        rr_qq_fn = f'{result_dir}/rr_qq{cf_tag}{cat_tag}_{nx}x{rlz_tag}.npy' #save for specific relatization, bc basis functions are rlz-dep
        if os.path.exists(rr_qq_fn) and not overwrite:
            # this might exist, e.g. if gradient tophot!
            print(f"RR & QQ already computed, loading in from {rr_qq_fn}")
            rr_proj, qq_proj, _ = np.load(rr_qq_fn, allow_pickle=True)
        else:
            rr_proj, qq_proj = compute_rr_qq_numeric(rr_qq_fn,random, L, r_edges, nprojbins, proj, proj_type, projfn=projfn, weights_r=weights_r, weight_type=weight_type, periodic=periodic)
        rr_proj, qq_proj = compute_rr_qq_numeric(rr_qq_fn,random, L, r_edges, nprojbins, proj, proj_type, 
                                                projfn=projfn, weights_r=weights_r, weight_type=weight_type, 
                                                periodic=periodic)

        qq_tag = '_qqnum'
        save_fn = f'{result_dir}/cf{cf_tag}{qq_tag}{cat_tag}_rlz{Nr}.npy'
        if os.path.exists(save_fn) and not overwrite:
            print(f"Already exists! ({save_fn}) continuing")
            continue
    
        print("Load in data")
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{Nr}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        assert L==Lx and L==Ly and L==Lz, f"Box sizes don't align! L: {L}, Lx: {Lx}, Ly: {Ly}, Lz: {Lz}"
        x, y, z, vx, vy, vz = data.T
        weights = np.array([np.ones(len(x)), x-loc_pivot[0], y-loc_pivot[1], z-loc_pivot[2]])
        #weights = np.array([np.ones(len(x)), x, y, z])
        print("N =",N)

        print("Computing xi")
        extra_dict.update({'r_edges': r_edges, 'nprojbins': nprojbins, 'proj_type': proj_type, 'projfn': projfn, 
                           'qq_analytic': qq_analytic, 'weight_type': weight_type, 'periodic': periodic})
        start = time.time()
        r, xi, amps = xi_proj_numeric(x, y, z, 
                                rx, ry, rz, rr_proj, qq_proj, L, r_edges,
                                nprojbins, proj_type, projfn=projfn, periodic=periodic,
                                weights=weights, weights_r=weights_r, weight_type=weight_type)
        end = time.time()

        #print("Done but not saved", save_fn)
        print(f"Saved result to {save_fn}")
        np.save(save_fn, [r, xi, amps, proj, extra_dict])
        print("Time:", end-start, "s")
            

def xi_proj_numeric(x, y, z, rx, ry, rz, rr_proj, qq_proj, L, r_edges, nprojbins, proj_type, projfn=None, weights=None, weights_r=None, weight_type=None, nthreads=24, periodic=False):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks

    print("computing dd...")
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                boxsize=L, periodic=periodic,
                weights1=weights, weight_type=weight_type)
    print("DD proj:", dd_proj)

    print("computing dr...")
    _, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, 
                    X2=rx, Y2=ry, Z2=rz,
                    proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
                    boxsize=L, periodic=periodic,
                    weights1=weights, weights2=weights_r, weight_type=weight_type)
    print("DR proj:", dr_proj)

    nd = len(x)
    nr = len(rx)
    rmin = min(r_edges)
    rmax = max(r_edges)
    r_cont = np.linspace(rmin, rmax, 1000)
    amps = compute_amps(nprojbins, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
    
    # for gradient, this will just return the tophat for now; we'll do the evaluation separately
    xi_proj = evaluate_xi(amps, r_cont, proj_type, rbins=r_edges, projfn=projfn)

    return r_cont, xi_proj, amps



def compute_rr_qq_numeric(rr_qq_fn, random, L, r_edges, nprojbins, proj, proj_type, projfn=None, weights_r=None, weight_type=None, nthreads=24, periodic=False):
    print("computing rr...")

    rx, ry, rz = random.T

    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    _, rr_proj, qq_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, rx, ry, rz,
            proj_type=proj_type, nprojbins=nprojbins, projfn=projfn,
            boxsize=L, periodic=periodic, weights1=weights_r, weight_type=weight_type)

    print("RR proj:", rr_proj)
    print("QQ proj:", qq_proj)   
    print(f"Saving RR and QQ to {rr_qq_fn}")
    np.save(rr_qq_fn, [rr_proj, qq_proj, proj])

    return rr_proj, qq_proj




if __name__=='__main__':
    main()
