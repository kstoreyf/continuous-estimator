import os
import numpy as np
import matplotlib.pyplot as plt
import glob 
import re

import nbodykit
import Corrfunc
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import bao

import read_lognormal as reader
import bao_utils


def main():
    boxsize = 750
    #cat_tag = f'_L{boxsize}_n2e-5_z057_patchy'
    #cat_tag = f'_L{boxsize}_n5e-5_z057_patchy'
    #cat_tag = f'_L{boxsize}_n1e-4_z057_patchy'
    cat_tag = f'_L{boxsize}_n2e-4_z057_patchy'
    proj = 'baoiter'
    cosmo_name = 'b17'
    #cosmo_name = 'planck'
    #cosmo_name = 'wmap9'
    cf_tag = f"_{proj}_cosmo{cosmo_name}_adaptive2"
    redshift = 0.57
    #realizations = range(1000)
    #realizations = range(30)
    realizations = [184]

    restart_unconverged = True # will restart any unconverged realizations - WARNING, IF FALSE WILL OVERWRITE ITERATIONS OF UNCONVERGED ONES
    convergence_threshold = 1e-5 #when to stop (fractional change)
    niter_max = 160 # stop after this many iterations if not converged 

    skip_converged = True
    qq_analytic = True
    nthreads = 24
    dalpha = 0.001
    #eta_start = 0.5 # theoretically 1 but need smaller for convergence issues
    k0 = 0.1

    print(Corrfunc.__file__)
    print(Corrfunc.__version__)

    cosmo = bao_utils.get_cosmo(cosmo_name)
    qq_tag = '' if qq_analytic else '_qqnum'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)

    for Nr in realizations:
        print(f"Realization {Nr}")

        if skip_converged:
            converged_fn = f'{result_dir}/cf{cf_tag}{qq_tag}_converged_niter*{cat_tag}_rlz{Nr}.npy'
            matches = glob.glob(converged_fn)
            if len(matches)>0:
                print("Already converged and saved to", matches[0])
                continue

        alpha_model_start = 1.0
        eta = 0.5
        biter = BAO_iterator(boxsize, cat_tag, Nr=Nr, cf_tag=cf_tag, qq_analytic=qq_analytic, nthreads=nthreads, cosmo=cosmo, redshift=redshift, alpha_model_start=alpha_model_start, dalpha=dalpha, k0=k0)# dfrac=dfrac)

        # initial parameters
        niter_start = 0
        if restart_unconverged:
            pattern = f"cf{biter.cf_tag}{qq_tag}_niter([0-9]+){biter.cat_tag}_rlz{biter.Nr}.npy"
            niters_done = []
            for fn in os.listdir(biter.result_dir):
                matches = re.search(pattern, fn)
                if matches is not None:
                    niters_done.append(int(matches.group(1)))
            
            if niters_done: # if matches found (list not empty), start from latest iter; otherwise, will start from zero
                niter_lastdone = max(niters_done) # load in last completed one
                start_fn = f'{biter.result_dir}/cf{biter.cf_tag}{qq_tag}_niter{niter_lastdone}{biter.cat_tag}_rlz{biter.Nr}.npy'
                res = np.load(start_fn, allow_pickle=True, encoding='latin1')
                _, _, amps, _, extra_dict = res
                #alpha_model_start = extra_dict['alpha_result']
                alpha_model_prev = extra_dict['alpha_model']
                C = amps[4]
                alpha_model_start = alpha_model_prev + eta*C*k0
                niter_start = niter_lastdone + 1
        
        print(f"Starting from iteration {niter_start}")
        # set up iterative procedure
        biter.load_catalogs()
        alpha_model = alpha_model_start
        #dalpha = dfrac*alpha_model

        #for niter in range(niter_start, niter_start+niters):
        niter = niter_start
        err = np.inf
        err_prev = np.inf
        alpha_result_prev = np.inf
        converged = False
        while (not converged) and (niter < niter_max):

            xi, amps = biter.bao_iterative(dalpha, alpha_model)
            C = amps[4]

            #alpha_result = alpha_model + dalpha*C
            #alpha_model_next = alpha_model + eta*dalpha*C
            alpha_result = alpha_model + C*k0
            extra_dict = {'r_edges': biter.rbins, 'nprojbins': biter.nprojbins, 
                          'proj_type': biter.proj_type, 'projfn': biter.projfn,
                          'alpha_start': alpha_model_start, 'alpha_model': alpha_model,
                          #'alpha_model_next': alpha_model_next, #for if iteration interrupted
                          'dalpha': dalpha, 'alpha_result': alpha_result, #should this be alpha_result?
                          'niter': niter}

            print(f'iter {niter}')
            print(f'alpha: {alpha_model}, dalpha: {dalpha}')
            print(f"C: {C}")
            #err = abs(alpha_model-alpha_result)/alpha_model
            err = (alpha_result - alpha_result_prev)/alpha_result
            if abs(err) < convergence_threshold:
                converged = True
            biter.save_cf(xi, amps, niter, extra_dict, converged=converged)

            # update alphas
            c1 = err>0
            c2 = err_prev>0
            if np.isfinite(err) and np.isfinite(err_prev) and (c1 != c2):
                # if the error switched sign, reduce eta
                eta *= 0.75
            print("Adaptive eta:", err, err_prev, eta)
            print(alpha_model,alpha_model + eta*C*k0)           
            alpha_model = alpha_model + eta*C*k0
           
            alpha_result_prev = alpha_result
            err_prev = err

            #dalpha = dfrac*alpha_next
            niter += 1
            
            print(f'NEW alpha: {alpha_model}, dalpha: {dalpha}')
            print(f'err: {err} (threshold: {convergence_threshold})')

        # Declare why stopped
        if niter==niter_max:
            print(f"hit max number of iterations, {niter_max}")
        if converged:
            print(f"converged after {niter} iterations with error {err} (threshold {convergence_threshold})")



class BAO_iterator:

    def __init__(self, boxsize, cat_tag, Nr=0, rmin=36.0, rmax=200.0, nbins=15, 
                 cf_tag='_baoiter', qq_analytic=True, nthreads=24, cosmo=None, 
                 redshift=0.0, bias=2.0, alpha_model_start=1.0, dalpha=0.01, k0=0.1):

        # input params
        self.boxsize = boxsize
        self.Nr = Nr
        if cosmo==None:
            print("No cosmo input, defaulting to Planck")
            self.cosmo = nbodykit.cosmology.Planck15
        else:
            self.cosmo = cosmo

        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.redshift = redshift

        # other params
        self.mumax = 1.0
        self.bias = bias
        self.k0 = k0
        self.weight_type=None
        self.periodic = True
        self.nthreads = nthreads
        self.nmubins = 1
        self.verbose = False
        self.proj_type = 'generalr'
        self.qq_analytic = qq_analytic

        # set up other data
        self.rbins = np.linspace(rmin, rmax, nbins+1)
        self.rbins_avg = 0.5*(self.rbins[1:]+self.rbins[:-1])
        self.rcont = np.linspace(rmin, rmax, 1000)

        self.cat_tag = cat_tag
        self.cat_dir = '../catalogs/lognormal'
        self.cf_tag = cf_tag
        self.projfn = f"../tables/bases{self.cat_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}_rlz{self.Nr}.dat"

        # set up result dir
        self.result_dir = '../results/results_lognormal{}'.format(self.cat_tag)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # write initial bases
        projfn_start = f"../tables/bases{self.cat_tag}{self.cf_tag}_r{self.rbins[0]}-{self.rbins[-1]}_z{self.redshift}_bias{self.bias}.dat"
        #dalpha = dfrac*alpha_model_start
        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_model':alpha_model_start, 'bias':self.bias}
        self.nprojbins, _ = bao.write_bases(self.rbins[0], self.rbins[-1], projfn_start, **kwargs)



    def save_cf(self, xi, amps, niter, extra_dict, converged=True):
        qq_tag = '' if self.qq_analytic else '_qqnum'
        conv_tag = '_converged' if converged else ''
        save_fn = f'{self.result_dir}/cf{self.cf_tag}{qq_tag}{conv_tag}_niter{niter}{self.cat_tag}_rlz{self.Nr}.npy'
        np.save(save_fn, [self.rcont, xi, amps, 'baoiter', extra_dict])
        print(f"Saved to {save_fn}")

    # NOT USED
    def load_true_cf(self, rmin, rmax):
        r_true, xi_true, _ = np.load('{}/cf_lin_true{}.npy'.format(self.cat_dir, self.cat_tag), 
            allow_pickle=True, encoding='latin1')
        xi_true = [xi_true[i] for i in range(len(r_true)) if rmin<=r_true[i]<rmax]
        r_true = [r_true[i] for i in range(len(r_true)) if rmin<=r_true[i]<rmax]
        return r_true, xi_true


    def load_catalogs(self):
        self.load_data()
        if not self.qq_analytic:
            self.load_random()


    def load_data(self):
        data_fn = f'{self.cat_dir}/cat{self.cat_tag}_lognormal_rlz{self.Nr}.bin'
        _, _, _, N, data = reader.read(data_fn) # first 3 are Lx, Ly, Lz
        self.x, self.y, self.z, _, _, _ = data.T
        self.nd = N
        self.weights = None


    def load_random(self):
        nx = 10
        rand_dir = '../catalogs/randoms'
        rand_fn = '{}/rand{}_{}x.dat'.format(rand_dir, self.cat_tag, nx)
        random = np.loadtxt(rand_fn)
        self.x_rand, self.y_rand, self.z_rand = random.T
        self.nr = random.shape[0]
        self.weights_rand = None


    def run_estimator_analytic(self):
        # TODO: make that can pass rbins as None to DDsmu for e.g. generalr when dont need!
        _, dd_proj, _ = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                        proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
                        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic, isa='fallback')

        volume = float(self.boxsize**3)
        rr_ana, qq_ana = qq_analytic(self.rmin, self.rmax, self.nd, volume, self.nprojbins, self.proj_type, rbins=self.rbins, projfn=self.projfn)
    
        numerator = dd_proj - rr_ana
        amps_periodic_ana = np.linalg.solve(qq_ana, numerator)
        print("AMPS:", amps_periodic_ana)
        xi_periodic_ana = evaluate_xi(amps_periodic_ana, self.rcont, self.proj_type, rbins=self.rbins, projfn=self.projfn)

        return xi_periodic_ana, amps_periodic_ana



    def run_estimator_numeric(self):

        _, dd_proj, _ = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                        proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
                        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)

        _, dr_proj, _ = DDsmu(0, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                            X2=self.x_rand, Y2=self.y_rand, Z2=self.z_rand, 
                            proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
                            verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)

        _, rr_proj, qq_proj = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x_rand, self.y_rand, self.z_rand,
               proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
               verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)
        
        print("nd nr", self.nd, self.nr)
        amps = compute_amps(self.nprojbins, self.nd, self.nd, self.nr, self.nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
        print("AMPS:", amps)
        xi_proj = evaluate_xi(self.nprojbins, amps, len(self.rcont), self.rcont, len(self.rbins)-1, self.rbins, self.proj_type, projfn=self.projfn)

        return xi_proj, amps


    # is this a good or an ugly way to do this toggle?
    def run_estimator(self):
        if self.qq_analytic:
            return self.run_estimator_analytic()
        else:
            return self.run_estimator_numeric()


    def bao_iterative(self, dalpha, alpha_model):

        kwargs = {'cosmo_base':self.cosmo, 'redshift':self.redshift, 'dalpha':dalpha, 'alpha_model':alpha_model, 'bias':self.bias, 'k0':self.k0}
        self.nprojbins, _ = bao.write_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)      
        xi, amps = self.run_estimator()
        
        return xi, amps



if __name__=="__main__":
    main()
