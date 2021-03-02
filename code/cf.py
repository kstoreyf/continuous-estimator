#!/usr/bin/env python
import os
import numpy as np
import time

import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import trr_analytic
from Corrfunc.bases import spline_bases
from Corrfunc.bases import bao_bases

import read_lognormal as reader
import bao_utils

from Corrfunc.theory.DD import DD
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from nbodykit.lab import *
import nbodykit




def main():

    print("Corrfunc version:", Corrfunc.__version__, Corrfunc.__file__)

    L = 750
    nbar_str = '5e-5'
    Nrealizations = 1
    #nbar_str = '2e-4'
    cat_tag = f'_L{L}_n{nbar_str}_z057_patchy'
    #cat_tag = f'_L{L}_n1e-4'
    periodic = True
    kwargs = {}
    
    #proj = 'theory'
    #binwidth = 6
    #cf_tag = f"_{proj}_bw{binwidth}"

    #proj = 'tophat'
    #binwidth = 6
    #cf_tag = f"_{proj}_bw{binwidth}_evalxitest"

    #proj = 'gradient'
    #binwidth = 10 #for bao, still need as dummy parameter
    #cf_tag = f"_{proj}_top_bw{binwidth}_nonperiodic"
    #kwargs = {'cat_tag':cat_tag, 'cf_tag_bao':'_baoiter_cosmob17_adaptive2'}
    #cf_tag = f"_{proj}_bao"

    # proj = 'piecewise'
    # binwidth = 10
    # cf_tag = f"_{proj}_bw{binwidth}_hangstill"

    proj = 'spline'
    kwargs = {'order': 3}
    binwidth = 12
    cf_tag = f"_{proj}{kwargs['order']}_bw{binwidth}_basetest"

    compute_xis(L, nbar_str, cat_tag, proj, cf_tag, binwidth=binwidth, kwargs=kwargs, Nrealizations=Nrealizations, trr_analytic=True, periodic=periodic)


def compute_xis(L, nbar_str, cat_tag, proj, cf_tag, binwidth=None, nbins=None, kwargs=None, Nrealizations=1000, overwrite=False, trr_analytic=True, nthreads=24, rmin=36.0, rmax=156.0, periodic=True):

    if trr_analytic and not periodic:
        print("Turning trr_analytic off, as periodic is False")
        trr_analytic = False

    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    rmin = rmin
    rmax = rmax
    assert bool(nbins) ^ bool(binwidth), "Set either nbins or binwidth (but not both)!"
    if nbins:   
        binwidth = (rmax-rmin)/float(nbins) 

    if proj in ['tophat', 'piecewise', 'spline']:
        r_edges = np.arange(rmin, rmax+binwidth, binwidth)
    else:
        r_edges = None

    if not (('gradient' in proj) and ('bao' in cf_tag)):
        proj_type, ncomponents, projfn, weight_type = get_proj_parameters(proj, r_edges=r_edges, cf_tag=cf_tag, **kwargs)
        assert not (proj_type=='gradient' and trr_analytic), "Can't use trr_analytic yet for gradient!"

    if not trr_analytic:
        # Load in randoms, will need for DR at least
        nx = 10
        rand_dir = '../catalogs/randoms'
        rand_fn = '{}/rand_L{}_n{}_{}x.dat'.format(rand_dir, L, nbar_str, nx)
        random = np.loadtxt(rand_fn)
        rx, ry, rz = random.T
        weights_r = np.array([np.ones(len(rx)), rx, ry, rz])
        # check if already have RR, compute if not
        rr_trr_fn = f'{result_dir}/rr_trr{cf_tag}{cat_tag}.npy'
        if os.path.exists(rr_trr_fn):
            print(f"RR & QQ already computed, loading in from {rr_trr_fn}")
            rr_proj, trr_proj, _ = np.load(rr_trr_fn, allow_pickle=True)
        else:
            rr_proj, trr_proj = compute_rr_trr_numeric(rr_trr_fn,random, L, r_edges, ncomponents, proj, proj_type, projfn=projfn, weights_r=weights_r, weight_type=weight_type, periodic=periodic)

    cat_dir = '../catalogs/lognormal'
    for Nr in range(Nrealizations):

        print(f"Realization {Nr}")
        # this needs to be in the loop for the gradient case
        if 'gradient' in proj and 'bao' in cf_tag:
            kwargs['Nr'] = Nr
            proj_type, ncomponents, projfn, weight_type = get_proj_parameters(proj, r_edges=r_edges, cf_tag=cf_tag, **kwargs)
            assert not (proj_type=='gradient' and trr_analytic), "Can't use trr_analytic yet for gradient!"

        trr_tag = '' if trr_analytic else '_trrnum'
        save_fn = f'{result_dir}/cf{cf_tag}{trr_tag}{cat_tag}_rlz{Nr}.npy'
        if os.path.exists(save_fn) and not overwrite:
            print(f"Already exists! ({save_fn}) continuing")
            continue
    
        print("Computing xi")
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{Nr}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        assert L==Lx and L==Ly and L==Lz, f"Box sizes don't align! L: {L}, Lx: {Lx}, Ly: {Ly}, Lz: {Lz}"
        x, y, z, vx, vy, vz = data.T
        print("N =",N)
        if proj_type=='gradient':
            weights = np.array([np.ones(len(x)), x, y, z])
        else:
            weights = None
        print("r_edges:", r_edges) 
        extra_dict = {'r_edges': r_edges, 'ncomponents': ncomponents, 'proj_type': proj_type, 'projfn': projfn, 'trr_analytic': trr_analytic}
        start = time.time()
        if "theory" in proj:
            xi = xi_theory(x, y, z, L, r_edges)
            r = 0.5*(r_edges[1:]+r_edges[:-1]) #use averages of r bins for theory xi
            amps = None
        else:
            if trr_analytic:
                r, xi, amps = xi_proj_analytic(x, y, z, L, r_edges,
                 ncomponents, proj_type, projfn=projfn)
            else:
                r, xi, amps = xi_proj_numeric(x, y, z, 
                                rx, ry, rz, rr_proj, trr_proj, L, r_edges,
                                ncomponents, proj_type, projfn=projfn, periodic=periodic,
                                weights=weights, weights_r=weights_r, weight_type=weight_type)
        end = time.time()

        #print("Done but not saved", save_fn)
        print(f"Saved result to {save_fn}")
        np.save(save_fn, [r, xi, amps, proj, extra_dict])
        print("Time:", end-start, "s")
            

def xi_theory(x, y, z, L, r_edges, nthreads=24):
    # this also assumes periodicity! 
    res = Corrfunc.theory.xi(L, nthreads, r_edges, x, y, z)
    res = np.array(res)
    xi = [rr[3] for rr in res]
    return xi


def xi_proj_analytic(x, y, z, L, r_edges, ncomponents, proj_type, projfn=None, nthreads=24):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    periodic = True # Periodic must be true for analytic computation!
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
              proj_type=proj_type, ncomponents=ncomponents, projfn=projfn,
              boxsize=L, periodic=periodic)
    #dd_proj = np.random.rand(ncomponents)
    rmin = min(r_edges)
    rmax = max(r_edges)
    volume = float(L**3)
    #nrbins = len(r_edges)-1
    nd = len(x)

    # works up to 100 thru here
    # hangs up to 15 when through next line
    print(rmin, rmax, nd, volume, ncomponents, r_edges, proj_type, projfn)
    rr_ana, trr_ana = trr_analytic(rmin, rmax, nd, volume, ncomponents, proj_type, rbins=r_edges, projfn=projfn)
    print(rr_ana)

    rcont = np.linspace(rmin, rmax, 1000)
    numerator = dd_proj - rr_ana
    amps_periodic_ana = np.linalg.solve(trr_ana, numerator)
    #xi_periodic_ana = evaluate_xi(nrbins, amps_periodic_ana, len(rcont), rcont, nrbins, r_edges, proj_type, projfn=projfn)
    xi_periodic_ana = evaluate_xi(amps_periodic_ana, rcont, proj_type, rbins=r_edges, projfn=projfn)

    print("DD proj:", dd_proj)
    print(numerator)
    print(amps_periodic_ana)
    print("First 10 xi vals:", xi_periodic_ana[:10])
    return rcont, xi_periodic_ana, amps_periodic_ana


def xi_proj_numeric(x, y, z, rx, ry, rz, rr_proj, trr_proj, L, r_edges, ncomponents, proj_type, projfn=None, weights=None, weights_r=None, weight_type=None, nthreads=24, periodic=True):
    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks

    print("computing dd...")
    _, dd_proj, _ = DDsmu(1, nthreads, r_edges, mumax, nmubins, x, y, z,
                proj_type=proj_type, ncomponents=ncomponents, projfn=projfn,
                boxsize=L, periodic=periodic,
                weights1=weights, weight_type=weight_type)
    print("DD proj:", dd_proj)


    print("computing dr...")
    _, dr_proj, _ = DDsmu(0, nthreads, r_edges, mumax, nmubins, x, y, z, 
                    X2=rx, Y2=ry, Z2=rz,
                    proj_type=proj_type, ncomponents=ncomponents, projfn=projfn,
                    boxsize=L, periodic=periodic,
                    weights1=weights, weights2=weights_r, weight_type=weight_type)
    print("DR proj:", dr_proj)

    nd = len(x)
    nr = len(rx)
    rmin = min(r_edges)
    rmax = max(r_edges)
    r_cont = np.linspace(rmin, rmax, 1000)
    amps = compute_amps(ncomponents, nd, nd, nr, nr, dd_proj, dr_proj, dr_proj, rr_proj, trr_proj)
    
    # for gradient, this will just return the tophat for now; we'll do the evaluation separately
    xi_proj = evaluate_xi(amps, r_cont, proj_type, rbins=r_edges, projfn=projfn)

    return r_cont, xi_proj, amps



def compute_rr_trr_numeric(rr_trr_fn, random, L, r_edges, ncomponents, proj, proj_type, projfn=None, weights_r=None, weight_type=None, nthreads=24, periodic=True):
    print("computing rr...")

    rx, ry, rz = random.T

    mumax = 1.0
    nmubins = 1
    verbose = False # MAKE SURE THIS IS FALSE otherwise breaks
    _, rr_proj, trr_proj = DDsmu(1, nthreads, r_edges, mumax, nmubins, rx, ry, rz,
            proj_type=proj_type, ncomponents=ncomponents, projfn=projfn,
            boxsize=L, periodic=periodic, weights1=weights_r, weight_type=weight_type)

    print("RR proj:", rr_proj)
    print("QQ proj:", trr_proj)   
    print(f"Saving RR and QQ to {rr_trr_fn}")
    np.save(rr_trr_fn, [rr_proj, trr_proj, proj])

    return rr_proj, trr_proj


def get_proj_parameters(proj, r_edges=None, cf_tag=None, **kwargs):
    proj_type = proj
    projfn = None
    weight_type = None
    if proj=="tophat" or proj_type=="piecewise":
        ncomponents = len(r_edges)-1
    elif proj=='spline':
        ncomponents = len(r_edges)-1
        proj_type = 'generalr'
        # cf_tag includes binwidth and order
        projfn = f"../tables/bases{cf_tag}_r{r_edges[0]}-{r_edges[-1]}_npb{ncomponents}.dat"
        spline_bases(r_edges[0], r_edges[-1], ncomponents, projfn, **kwargs)
    elif proj=='gradient':
        if 'top' in cf_tag:
            ncomponents = 4*(len(r_edges)-1)
        elif 'bao' in cf_tag:
            projfn, ncomponents = bao_utils.get_gradient_bao_params(**kwargs)
        weight_type = 'pair_product_gradient'
    elif proj=='theory':
        proj_type = None
        ncomponents = None
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    return proj_type, ncomponents, projfn, weight_type





if __name__=='__main__':
    main()
