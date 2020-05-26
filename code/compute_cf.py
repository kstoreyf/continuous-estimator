#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from nbodykit.lab import *
import nbodykit
import Corrfunc
from Corrfunc.theory.DD import DD
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

import plotter
import corrfuncproj

from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi

import read_lognormal as reader


nthreads = 24
color_dict = {'True':'black', 'tophat':'blue', 'LS': 'orange', 'piecewise':'red', 'standard': 'orange'}


def main():
    print("Go!")
    multi()


def multi():
    nrealizations = 15
    #seeds = [10]
    seeds = np.arange(0,nrealizations)
    boxsize = 750
    #nbar_str = '3e-4'
    nbar_str = '1e-4'
    mock_tag = ''

    nbins = 22
    projs = ['tophat']
    proj_tags = ['tophat_hangcheck']
    #projs = ['bao']
    #proj_tags = ['bao_alpha1.01']
    # for bao only
    #kwargs = {'cosmo_base':nbodykit.cosmology.Planck15, 'redshift':0}
    cosmo = nbodykit.cosmology.Planck15
    
    #py_str = '_py2'
    compute_standard = False # also compute the standard estimator
    overwrite = True
    overwrite_rr = False
    py_str = ''
    if 'py2' in py_str:
        allow_pickle = False
    else:
        allow_pickle = True
    
    # for dcosmo only
    # TODO: make sure cosmo is aligned with sim loaded in
    kwargs = {}
    #kwargs = {'params':['Omega_cdm', 'Omega_b', 'h'], 'cosmo_base':nbodykit.cosmology.Planck15, 'redshift':0}

    #cat_tag = '_L{}_nbar{}{}{}'.format(boxsize, nbar_str, mock_tag, py_str)
    cat_tag = '_L{}_n{}{}{}'.format(boxsize, nbar_str, mock_tag, py_str)

    tagstr = ','.join(proj_tags)
    print("Running box {} for projections {}".format(cat_tag[1:], tagstr))
    
    #cat_dir = '../catalogs/cats_lognormal{}'.format(cat_tag)
    #cat_dir = '../../byebyebias/catalogs/cats_lognormal{}'.format(cat_tag) # WORKS
    cat_dir = '../catalogs/lognormal'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log = False
    
    # randoms
    #rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, cat_tag)
    #random = np.loadtxt(rand_fn)
    #nr = random.shape[0]
    #randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, cat_tag)        
    #randomsky = np.loadtxt(randsky_fn)

    print("Set up bins")
    if log:
        rmin = 1
    else:
        rmin = 40
    rmax = 150
    #nbins = 16
   
    # Load True CF (input to catalog) 
    rbins = np.linspace(rmin, rmax, nbins+1)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    #r_lin, _, _ = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true_cont1000', cat_tag), allow_pickle=allow_pickle)
    # TODO: MAKE SIM MORE BINS, THIS IS HACKY
    #r_lin = np.linspace(min(r_lin), max(r_lin), 1000)
    r_lin = np.linspace(rmin, rmax, 1000)
    if log:
        rbins_log = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
        rbins_avg_log = 10 ** (0.5 * (np.log10(rbins_log)[1:] + np.log10(rbins_log)[:-1]))
        r_log, _, _ = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))

    ### STANDARD RANDOMS
    ### check if random counts already exist for tag, if not then count
    if compute_standard:
        save_rrstandard_fn = '{}/rr_qq_standard_lin{}.npy'.format(result_dir, cat_tag)
        if not os.path.isfile(save_rrstandard_fn) or overwrite_rr:
            print("Computing randoms for standard")
            rr = counts_corrfunc_auto(random, rbins, boxsize)
            np.save(save_rrstandard_fn, rr)
            print("Saved standard randoms to {}".format(save_rrstandard_fn))
        else:
            print("Loading standard randoms from {}".format(save_rrstandard_fn))
            rr = np.load(save_rrstandard_fn, allow_pickle=allow_pickle)

    ### PROJECTION RANDOMS
    ### check if random counts already exist for tag, if not then count
    #rr_projs, qq_projs = [], []
    #for i in range(len(projs)):
    #    save_rrproj_fn = '{}/rr_qq_proj_lin_{}{}.npy'.format(result_dir, proj_tags[i], cat_tag)
    #    if not os.path.isfile(save_rrproj_fn) or overwrite_rr:
    #        print("Computing randoms for projection {}".format(proj_tags[i]))
    #        rr_proj, qq_proj = counts_cf_proj_auto(randomsky, rbins, r_lin, projs[i], qq=True, **kwargs)
    #        np.save(save_rrproj_fn, [rr_proj, qq_proj])
    #        print("Saved projection randoms to {}".format(save_rrproj_fn))
    #    else:
    #        print("Loading projection randoms from {}".format(save_rrproj_fn))
    #        rr_proj, qq_proj = np.load(save_rrproj_fn, allow_pickle=allow_pickle)
    #    rr_projs.append(rr_proj)
    #    qq_projs.append(qq_proj)

    ### COMPUTE CFs
    ### for each seed for each projection
    for seed in seeds:

        # data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
        # data = np.loadtxt(data_fn)
        # nd = data.shape[0]
        # datasky_fn = '{}/catsky_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
        # datasky = np.loadtxt(datasky_fn)
        fn = f'{cat_dir}/cat{cat_tag}_lognormal_rlz{seed}.bin'
        Lx, Ly, Lz, N, data = reader.read(fn)
        x, y, z, vx, vy, vz = data.T
        pos = np.array([x, y, z]).T
        ra, dec, cz = to_sky(pos, cosmo)
        print("to skyed")
        datasky = [ra, dec, cz]
        print("um")
        print(datasky)
        datasky = np.array(datasky).T
        print("donesky")

        if compute_standard:
            save_standard_fn = '{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, 'standard', cat_tag, seed)
            if not os.path.isfile(save_standard_fn) or overwrite:
                dd = counts_corrfunc_auto(data, rbins, boxsize)
                dr = counts_corrfunc_cross(data, random, rbins, boxsize)
                xi_stan = compute_cf(dd, dr, rr, nd, nr, 'ls')
                np.save('{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, 'standard', cat_tag, seed), [rbins_avg, xi_stan, 'standard'])
            else:
                print("{} already exists!".format(save_standard_fn))
    
        for i in range(len(projs)):
            print("COMPUTING: Proj {}, cat_tag {}, seed {}".format(proj_tags[i], cat_tag, seed))
            save_fn = '{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, proj_tags[i], cat_tag, seed)
                   
            if not os.path.isfile(save_fn) or overwrite:
                start = time.time()
                proj = projs[i]
                dd_proj = counts_cf_proj_auto(datasky, rbins, r_lin, proj, qq=False, **kwargs)
                #dr_proj = counts_cf_proj_cross(datasky, randomsky, rbins, r_lin, proj, **kwargs)
                #r_proj, xi_proj, amps_proj = compute_cf_proj(dd_proj, dr_proj, rr_projs[i], qq_projs[i], nd, nr, rbins, r_lin, proj, **kwargs)
                print("dddddd proj done")
                #print("Amplitudes:", amps_proj)
                #np.save(save_fn, [r_proj, xi_proj, amps_proj, proj])
                end = time.time()
                print("Saved to {}".format(save_fn))
                print("TIME FOR PROJCF: {} s".format(end-start))
            else:
                print("{} already exists!".format(save_fn))


def counts_corrfunc_auto(cat, rbins, boxsize):
    x, y, z = cat.T
    periodic = False
    print('Starting auto counts')
    s = time.time()
    dd = DD(1, nthreads, rbins, X1=x, Y1=y, Z1=z,
               periodic=periodic, boxsize=boxsize)
    dd = np.array([x[3] for x in dd])
    print(dd)
    print('time: {}'.format(time.time()-s))
    return dd


def counts_corrfunc_cross(data, random, rbins, boxsize):
    s = time.time()
    periodic = False
    datax, datay, dataz = data.T
    randx, randy, randz = random.T
    print("Starting cross counts")
    dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, X2=randx, Y2=randy, Z2=randz, boxsize=boxsize)
    dr = np.array([x[3] for x in dr])
    print(dr)
    print('time: {}'.format(time.time()-s))
    return dr


def counts_corrfunc_3d(data, random, rbins, boxsize):
    print(data.shape)
    datax, datay, dataz = data.T
    randx, randy, randz = random.T
    
    periodic = True
    print('Starting counts')
    s = time.time()
    dd = DD(1, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, boxsize=boxsize)
    dd = np.array([x[3] for x in dd])
    print('DD:',dd)
    print('time: {}'.format(time.time()-s))
    s = time.time()
    dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, X2=randx, Y2=randy, Z2=randz, boxsize=boxsize)
    dr = np.array([x[3] for x in dr])
    print('DR:',dr)
    print('time: {}'.format(time.time()-s))
    s = time.time()
    rr = DD(1, nthreads, rbins, randx, randy, randz,
                periodic=periodic, boxsize=boxsize)
    rr = np.array([x[3] for x in rr])
    print('RR:',rr)
    print('time: {}'.format(time.time()-s))

    return dd, dr, rr


def compute_cf(dd, dr, rr, nd, nr, est):

    dd = np.array(dd).astype(float)
    dr = np.array(dr).astype(float)
    rr = np.array(rr).astype(float)

    fN = float(nr)/float(nd)
    if est=='ls':
        return (dd * fN**2 - 2*dr * fN + rr)/rr
    elif est=='natural':
        return fN**2*(dd/rr) - 1
    elif est=='dp':
        return fN*(dd/dr) - 1
    elif est=='ham':
        return (dd*rr)/(dr**2) - 1
    else:
        exit("Estimator '{}' not recognized".format(est))


def counts_cf_proj_auto(cat, rbins, r_cont, proj, qq=False, **kwargs):
    cosmo = 1 #doesn't matter bc passing cz, but required

    ra, dec, cz = cat.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

    mumax = 1.0 #max of cosine
    weights = None
    nproc = nthreads
    res = corrfuncproj.counts_smu_auto(ra, dec, cz, 
                                  rbins, mumax, cosmo, nproc=nproc,
                                  weights=weights,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn, qq=qq)
    if qq:
        dd_proj, dd_res_corrfunc, dd_projt = res
        return dd_proj, dd_projt
    else: 
        dd_proj, dd_res_corrfunc = res
        return dd_proj
    

def counts_cf_proj_cross(data, random, rbins, r_cont, proj, **kwargs):
    cosmo = 1 #doesn't matter bc passing cz, but required
    
    datara, datadec, datacz = data.T
    randra, randdec, randcz = random.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

    mumax = 1.0 #max of cosine
    weights_data = None
    weights_rand = None
    nproc = nthreads
    res = corrfuncproj.counts_smu_cross(datara, datadec,
                                  datacz, randra, randdec,
                                  randcz, rbins, mumax, cosmo, nproc=nproc,
                                  weights_data=weights_data, weights_rand=weights_rand,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn)

    dr_proj, dr_res_corrfunc = res
    return dr_proj



def counts_cf_proj(data, random, rbins, r_cont, proj, **kwargs):

    #cosmo = LambdaCDM(H0=70, Om0=0.25, Ode0=0.75)
    cosmo = 1 #doesn't matter bc passing cz, but required

    datara, datadec, datacz = data.T
    randra, randdec, randcz = random.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

    mumax = 1.0 #max of cosine
    weights_data = None
    weights_rand = None
    nproc = nthreads
    res = corrfuncproj.counts_smu(datara, datadec,
                                  datacz, randra, randdec,
                                  randcz, rbins, mumax, cosmo, nproc=nproc,
                                  weights_data=weights_data, weights_rand=weights_rand,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn)

    dd, dr, rr, qq, dd_orig, dr_orig, rr_orig = res

    return dd, dr, rr, qq


def compute_cf_proj(dd, dr, rr, qq, nd, nr, rbins, r_cont, proj, **kwargs):
    if 'kernel' in proj:
        r_proj = 0.5*(rbins[1:]+rbins[:-1])
        print('rproj', len(r_proj))
        xi_proj = compute_cf(dd, dr, rr, nd, nr, 'ls')
    else:
        r_proj = r_cont
        proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)
        # Note: dr twice because cross-correlations will be possible
        amps = compute_amps(nprojbins, nd, nd, nr, nr, dd, dr, dr, rr, qq)
        print('Computed amplitudes')

        amps = np.array(amps)

        rbins = np.array(rbins)
        xi_proj = evaluate_xi(nprojbins, amps, len(r_cont), r_cont, len(rbins)-1, rbins, proj_type, projfn=projfn)
        print("Computed xi")
    return r_proj, xi_proj, amps



def get_proj_parameters(proj, rbins=None, **kwargs):
    proj_type = proj
    projfn = None
    if proj=="tophat" or proj_type=="piecewise":
        nprojbins = len(rbins)-1
    elif proj=="powerlaw":
        nprojbins = 3
    elif proj=='generalr':
        nprojbins = 3
        #projfn = "/home/users/ksf293/vectorizedEstimator/tables/dcosmos_rsd_norm.dat"
        projfn = "/home/users/ksf293/vectorizedEstimator/tables/dcosmos_norm.dat"
    elif proj=='dcosmo':
        proj_type = 'generalr'
        #params = ['Omega_cdm', 'Omega_b', 'h']
        projfn = '../tables/dcosmo.dat'
        nprojbins, _ = dcosmo.write_bases(rbins[0], rbins[-1], projfn, **kwargs)
    elif proj=='linear_spline':
        nprojbins = len(rbins)-1
        proj_type = 'generalr'
        projfn = '../tables/linear_spline.dat'
        spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, 1, projfn)
    elif proj=='quadratic_spline':
        nprojbins = len(rbins)-1
        proj_type = 'generalr'
        projfn = '../tables/quadratic_spline.dat'
        spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, 2, projfn)
    elif proj=='cubic_spline':
        nprojbins = len(rbins)-1
        proj_type = 'generalr'
        projfn = '../tables/cubic_spline.dat'
        spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, 3, projfn) 
    elif proj=='gaussian_kernel':
        nprojbins = len(rbins)-1
        projfn = '../tables/gaussian_kernel.dat'
        ncont = len(rbins)-1
        nprojbins, _ = kernel.write_bases(rbins[0], rbins[-1], projfn, ncont=ncont)
    elif proj=='bao':
        proj_type = 'generalr'
        projfn = '../tables/bao.dat'
        print(kwargs)
        nprojbins, _ = bao.write_bases(rbins[0], rbins[-1], projfn, **kwargs) 
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    print("nprojbins:", nprojbins)
    return proj_type, nprojbins, projfn


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
    #return ra, dec, z
    #return np.array([ra, dec, z])


if __name__=='__main__':
    main()
