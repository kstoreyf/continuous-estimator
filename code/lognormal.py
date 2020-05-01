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




def main():
    realizations()

def realizations():
    boxsize = 750
    nbar_str = '3e-4'
    #nrealizations = 11
    #seeds = np.arange(nrealizations)
    seeds = [999]
    tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    
    print("Making lognormal mocks for {}".format(tag))
    
    cat_dir = '../catalogs/cats_lognormal{}'.format(tag)
    if not os.path.isdir(cat_dir):
        os.makedirs(cat_dir)

    data_fn = '{}/cat_lognormal{}.dat'.format(cat_dir, tag)
    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, tag)
    datasky_fn = '{}/catsky_lognormal{}.dat'.format(cat_dir, tag)
    randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, tag)
    pk_fn = '{}/pk_cont1000{}.dat'.format(cat_dir, tag)
    cf_lin_fn = '{}/cf_lin_{}{}.npy'.format(cat_dir, 'true_cont1000', tag)
    cf_log_fn = '{}/cf_log_{}{}.npy'.format(cat_dir, 'true_cont1000', tag)
    nbar = float(nbar_str)
    boxsize = float(boxsize)

    redshift = 0
    cosmo = cosmology.Planck15
    print("Generating power spectrum")
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

    pk(Plin, saveto=pk_fn)
    cf(Plin, log=False, saveto=cf_lin_fn)
    cf(Plin, log=True, saveto=cf_log_fn)

   # if not os.path.isfile(rand_fn):
   #     random = generate_random(nbar, boxsize, savepos=rand_fn)
   #     randomsky = to_sky(random['Position'], cosmo, savepos=randsky_fn)

   # for seed in seeds:
   #     data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(cat_dir, tag, seed)
   #     datasky_fn = '{}/catsky_lognormal{}_seed{}.dat'.format(cat_dir, tag, seed)
   #     if not os.path.isfile(data_fn):
   #         data = generate_data(nbar, boxsize, Plin, seed=seed, savepos=data_fn)
   #         datasky = to_sky(data['Position'], cosmo, savepos=datasky_fn)


def pk(Plin, saveto=None, ncont=1000):
    print("Power spectrum and correlation function")
    k = np.logspace(-3, 2, ncont)
    Pk = Plin(k)
    if saveto:
        np.save(saveto, [k, Pk])
    return k, Pk


def cf(Plin, log=False, rmin=1, rmax=150, saveto=None, ncont=1000):
    if log:
        r = np.logspace(np.log10(rmin), np.log10(rmax), ncont)
    else:
        r = np.linspace(rmin, rmax, ncont)
    CF = nbodykit.cosmology.correlation.CorrelationFunction(Plin)
    xi = CF(r)#, smoothing=0.0, kmin=10**-2, kmax=10**0)
    if saveto:
        np.save(saveto, [r, xi, 'true'])
    return r, xi


def generate_data(nbar, boxsize, Plin, seed=42, savepos=None):
    b1 = 1.0
    print("Making data catalog")
    s = time.time()
    data = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=boxsize, Nmesh=256, bias=b1, seed=seed)
    print('time: {}'.format(time.time()-s))
    nd = data.csize
    print("Data: {}".format(nd))
    if savepos:
        datapos = get_positions(data)
        np.savetxt(savepos, np.array(datapos).T)
    return data


def generate_random(nbar, boxsize, seed=43, savepos=None):
    print("Making random catalog")
    s = time.time()
    random = nbodykit.source.catalog.uniform.UniformCatalog(10*nbar, boxsize, seed=seed)
    print('time: {}'.format(time.time()-s))
    nr = random.csize
    print("Random: {}".format(nr))
    if savepos:
        randompos = get_positions(random)
        np.savetxt(savepos, np.array(randompos).T)
    return random


def single():
    boxsize = 750
    nbar_str = '3e-4'
    tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    cat_fn = '{}/cat_lognormal{}.dat'.format(cat_dir, tag)
    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, tag)
    catsky_fn = '{}/catsky_lognormal{}.dat'.format(cat_dir, tag)
    randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, tag)
    pk_fn = '{}/pk{}.dat'.format(cat_dir, tag)

    nbar = float(nbar_str)
    boxside = float(boxsize)

    redshift = 0
    cosmo = cosmology.Planck15
    print("Generating power spectrum")
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 1.0

    print("Power spectrum and correlation function")
    k = np.logspace(-3, 2, 300)
    Pk = Plin(k)
    np.save(pk_fn, [k, Pk])
    rmin = 1
    rmax = 150
    r_lin = np.linspace(rmin, rmax, 300)
    r_log = np.logspace(np.log10(rmin), np.log10(rmax), 300)
    CF = nbodykit.cosmology.correlation.CorrelationFunction(Plin)
    xi_log = CF(r_log)#, smoothing=0.0, kmin=10**-2, kmax=10**0)
    xi_lin = CF(r_lin)
    np.save('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', tag), [r_lin, xi_lin, 'true'])
    np.save('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', tag), [r_log, xi_log, 'true'])

    print("Making data catalog")
    s = time.time()
    data = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=boxsize, Nmesh=256, bias=b1, seed=42)
    print('time: {}'.format(time.time()-s)) 
    nd = data.csize

    print("Making random catalog")
    s = time.time()
    random = nbodykit.source.catalog.uniform.UniformCatalog(10*nbar, boxsize, seed=43)
    print('time: {}'.format(time.time()-s))
    nr = random.csize
    print(nd, nr)   

    datasky = to_sky(data['Position'], cosmo)
    randomsky = to_sky(random['Position'], cosmo)

    np.savetxt(catsky_fn, np.array(datasky).T)
    np.savetxt(randsky_fn, np.array(randomsky).T) 

    data = get_positions(data)
    random = get_positions(random)

    np.savetxt(cat_fn, np.array(data).T)
    np.savetxt(rand_fn, np.array(random).T)

    np.savetxt(catsky_fn, np.array(datasky).T)
    np.savetxt(randsky_fn, np.array(randomsky).T)


def to_sky(pos, cosmo, velocity=None, rsd=False, comoving=True, savepos=None):
    if rsd:
        if velocity is None:
            raise ValueError("Must provide velocities for RSD! Or set rsd=False.")
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    else:
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo)
    if comoving:
        z = cosmo.comoving_distance(z)

    ra = ra.compute().astype(float)
    dec = dec.compute().astype(float)

    pos = [ra, dec, z]
    if savepos:
        np.savetxt(savepos, np.array(pos).T)

    return pos


def get_positions(cat):
    catx = np.array(cat['Position'][:,0]).astype(float)
    caty = np.array(cat['Position'][:,1]).astype(float)
    catz = np.array(cat['Position'][:,2]).astype(float)
    return catx, caty, catz

if __name__=='__main__':
    main()
