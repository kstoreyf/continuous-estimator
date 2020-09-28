import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d

import utils


color_dict = {'True':'black', 'tophat':'blue', 'standard': 'orange', 'piecewise':'crimson', 'linear spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson'}

ctopmany = '#33BBEE'
ctopmid = '#3547F1'
ctopfew = '#0077BB'
cspline = '#EE3377'
cbao = '#228833'
ctheory = '#808080'

base_colors = ['magenta', 'red', 'orange', 'green', 'blue']
base_colors = ['#332288', '#88CCEE', '#999933', '#CC6677', '#AA4499'] #from muted colorscheme here: https://personal.sron.nl/~pault/#sec:qualitative
#base_names = ['a1', 'a2', 'a3', 'Bsq', 'C']
base_names = [r'$\xi^{mod}(\alpha_\mathrm{guess} s)$', r'$k_0 \frac{\dd \xi^{mod}(\alpha_\mathrm{guess} s)}{\dd \alpha}$', r'$\frac{k_1}{s^2}$', r'$\frac{k_2}{s}$', r'$k_3$']

def plot_cf_cont(rs, cfs, r_true, cf_true, labels, colors, alphas=None, saveto=None,
            log=False, err=False, error_regions=None, xlim=None, errlim=None, cont=True, 
            label_rmse=True, show_legend=True, bases=None):

    if alphas is None:
        alphas = np.ones(len(colors))

    if err is not None and bases is None:
        ax_main = 0
        ax_err = 1
        fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0.1)
    elif err is not None and bases is not None:
        ax_bases = 0
        ax_main = 1
        ax_err = 2
        fig, ax = plt.subplots(3, 1, figsize=(8,10), gridspec_kw={'height_ratios': [1, 2, 1]})
        plt.subplots_adjust(wspace=0, hspace=0.2)
    else:
        ax_main = 0
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        ax = [ax]
    
    if cont:
        lss = ['-']*len(rs)
        marker = 'None'
    else:
        lss = ['None']*len(rs)
        marker = 'd'

    if xlim is None:
        if log:
            xmin = 0
        else:
            xmin = 40
        xmax = max(np.array(rs).flatten())
    else:
        xmin, xmax = xlim

    r_t = np.array([r_true[k] for k in range(len(r_true)) if xmin<=r_true[k]<xmax])
    cf_t = np.array([cf_true[k] for k in range(len(r_true)) if xmin<=r_true[k]<xmax])
   
    ax[ax_main].axhline(0, color='silver', ls='-')
    ax[ax_main].plot(r_t, cf_t, color='k', label='True', ls='-', lw=1.5, zorder=1000)
    
    offset = 0
    for j in range(len(rs)):
         
        r = np.array([rs[j][k] for k in range(len(rs[j])) if xmin<=rs[j][k]<xmax])
        cf = np.array([cfs[j][k] for k in range(len(rs[j])) if xmin<=rs[j][k]<xmax])

        if error_regions is not None:
            lower = error_regions[j][0]
            upper = error_regions[j][1]
            lower = np.array([lower[k] for k in range(len(rs[j])) if xmin<=rs[j][k]<xmax])
            upper = np.array([upper[k] for k in range(len(rs[j])) if xmin<=rs[j][k]<xmax])
        #cf = 1 + cf
        #cf = r**2 * cf
        if len(rs[j])==len(r_true) and abs(max(rs[j])-max(r_true))<0.01:
            #marker = None
            #ls = '-'
            rs_are_same = True
            rmserr = rmse(cf, cf_t)
            #print(labels[j], "RMSE: {:.2e}".format(rmserr))
            if label_rmse:
                label = '{} (rmse: {:.2e})'.format(labels[j], rmserr)
            else:
                label = labels[j]
        else:
            rs_are_same = False
            #marker = 'o'
            #ls = 'None'
            label = labels[j]
        ax[ax_main].plot(r, cf, color=colors[j], alpha=alphas[j], label=str(label), marker=marker, ls=lss[j], lw=2.5)

        if error_regions is not None:
            if cont:
                ax[ax_main].fill_between(r, lower,  upper, color=colors[j], alpha=0.2)
                ax[ax_main].plot(r, lower, color=colors[j], ls=lss[j])
                ax[ax_main].plot(r, upper, color=colors[j], ls=lss[j])
            else:
                ax[ax_main].errorbar(r+offset, cf, yerr=[cf-lower, upper-cf], color=colors[j], ls='None', alpha=0.5)

        if err:
            #ax[1].plot(r, (cf-cf_t)/cf_t, color=colors[j], alpha=alphas[j])
            if not rs_are_same:
                cf_t_func = interp1d(r_true, cf_true, kind='cubic')
                cf_t = cf_t_func(r)

            ax[ax_err].plot(r, cf-cf_t, color=colors[j], alpha=alphas[j], marker=marker, ls=lss[j], lw=2.5)
            # for now let's not plot error in residual
            #if cont:
                #ax[1].fill_between(r, upper-cf_t, lower-cf_t, color=colors[j], alpha=0.2)
                #ax[1].plot(r, lower-cf_t, color=colors[j], ls=lss[j])
                #ax[1].plot(r, upper-cf_t, color=colors[j], ls=lss[j])
            #else:
            #    ax[1].errorbar(r+offset, cf-cf_t, yerr=[cf-lower, upper-cf], color=colors[j], ls='None', alpha=0.5)
            #ax[1].plot(r, cf/cf_t, color=colors[j], alpha=alphas[j])

        if bases is not None:
            base = bases[j]
            rbase = base[:, 0]
            base = np.array([base[k] for k in range(len(rbase)) if xmin<=rbase[k]<xmax])
            rb = np.array([rbase[k] for k in range(len(rbase)) if xmin<=rbase[k]<xmax])
            ax[ax_bases].set_ylim(-0.1, 1.1)
            print(labels[j])
            rescale_by = [2.5, 3.0, 2.0, 3.0, 1.0]
            for bb in range(1, base.shape[1]):
                nz = np.nonzero(base[:,bb])[0]
                if len(nz)==0:
                    continue
                minnz = min(nz)
                maxnz = max(nz)
                if min(nz) != 0:
                    minnz -= 1
                if max(nz) != len(base[:,bb]-1):
                    maxnz += 1
                base_nonzero = base[:,bb][minnz:maxnz]
                rb_nonzero = rb[minnz:maxnz]
                if "BAO" in labels[j]:
                    base_normed = rescale_by[bb-1]*base_nonzero
                else:
                    base_normed = base_nonzero
                ax[ax_bases].plot(rb_nonzero, base_normed, color=colors[j], alpha=alphas[j], marker=marker, ls=lss[j], lw=2.5)


    ax[ax_main].set_ylabel(r'$\xi(r)$')
    ax[ax_main].set_xlim(xmin, xmax)
    
    if log:
        ax[ax_main].set_xscale('log')
        ax[ax_main].set_yscale('log')
        if err:
            ax[ax_err].set_xscale('log')

    if err:
        ax[ax_err].axhline(0, color='k', lw=1.5, zorder=1000)
        ax[ax_err].set_xlim(xmin, xmax)

        if bases is not None:
            ax[ax_bases].set_xlim(xmin, xmax)
            ax[ax_bases].set_ylabel(r'bases')

        ax[ax_err].set_ylabel(r'$\xi(r)$ - $\xi_{true}(r)$')
        if errlim:
            ax[ax_err].set_ylim(errlim[0], errlim[1])

    ax[-1].set_xlabel(r'r (h$^{-1}$ Mpc)')

    if show_legend:
        ax[ax_main].legend()
    if saveto:
        plt.savefig(saveto)
    return ax
    

def plot_cf(rs, cfs, ests, cftrue, r_cont, cftrue_cont, saveto=None,
            log=False, err=False, zoom=False):

    colors = ['r','g','b','m']

    if err:
        fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]})
    else:
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        ax = [ax]

    for j in range(len(ests)):
        mean = np.mean(cfs[:,j], axis=0)
        std = np.std(cfs[:,j], axis=0)

        offset = 10**(np.log10(rs[j])+np.log10(0.02*j))
        ax[0].errorbar(rs[j]+offset, mean, yerr=std, capsize=1, color=colors[j], label=ests[j])

        if err:
            if 'proj' in ests[j]:
                ax[1].plot(rs[j], (mean-cftrue_cont)/cftrue_cont, color=colors[j])
            else:
                ax[1].plot(rs[j], (mean-cftrue)/cftrue, color=colors[j])



    ax[0].plot(r_cont, cftrue_cont, color='k', label='true')

    ax[0].set_xlabel('r')
    ax[0].set_ylabel(r'$\xi(r)$')

    ax[0].set_xscale('log')
    if zoom:
        ax[0].set_ylim(-0.05, 0.05)

    if log:
        ax[0].set_yscale('log')

    if err:
        ax[1].axhline(0, color='k')
        ax[1].set_xlabel('r')
        ax[1].set_ylabel(r'($\xi-\xi_{true})/\xi_{true}$')
        ax[1].set_xscale('log')
        ax[1].set_ylim(-0.5, 0.5)

    ax[0].legend()

    if saveto:
        plt.savefig(saveto)


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


def plot_sim(data, random, boxsize, zrange=None, saveto=None):
    plt.figure()
    if zrange:
        data = np.array([d for d in data.T if zrange[0]<=d[2]<zrange[1]])
        data = data.T
        random = np.array([r for r in random.T if zrange[0]<=r[2]<zrange[1]])
        random = random.T
    plt.scatter(random[0], random[1], s=1, color='cyan', label='random')
    if len(data)>0:
        plt.scatter(data[0], data[1], s=1, color='red', label='data')
    plt.legend(loc='upper right',framealpha=1)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    plt.xlim(0, boxsize)
    plt.ylim(0, boxsize)
    plt.xlabel(r'$x$ (h$^{-1}$Mpc)')
    plt.ylabel(r'$y$ (h$^{-1}$Mpc)')
    if saveto:
        plt.savefig(saveto)


def plot_cf_err(rs, cf_arrs, r_true, cf_true, labels, colors, err=False, xlim=None, errlim=None, cont=True, bases=None):
    
    if np.array(rs).ndim == 1:
        rs = [rs]
    if np.array(cf_arrs).ndim == 2:
        cf_arrs = np.array([cf_arrs])
    
    cfs_mean = []
    error_regions = []
    for cfs in cf_arrs:
        mean = np.mean(cfs, axis=0)
        cfs_mean.append(mean)
        std = np.std(cfs, axis=0)
        error_regions.append([mean-std, mean+std])
    
    ax = plot_cf_cont(rs, cfs_mean, r_true, cf_true, labels, colors, error_regions=error_regions, 
                         err=err, xlim=xlim, errlim=errlim, cont=cont, bases=bases)
    return ax


def plot_continuous(cat_tag, cf_tags, Nrealizations=100, colors=None, labels=None, err=True, errlim=None, 
                    cont=True, show_bases=True, xlim=None, peak_fit=False, bws=[], r_widths=[], r_max_true=None, b1=2.0):
    
    if colors is None:
        colors = ['lime','blue', 'cyan', 'magenta', 'purple']
    if labels is None:
        labels = [f"{tag.split('_')[1]}, bin width {tag.split('bw')[-1]}" for tag in cf_tags]
    if xlim is None:
        xlim = [40.0, 148.0]
    
    cat_dir = '../catalogs'
    result_dir = '../results/results_lognormal{}'.format(cat_tag)

    rs = []
    r_arrs = []
    cf_arrs = []
    bases = None
    if show_bases:
        bases = []

    for i in range(len(cf_tags)):
        cf_tag = cf_tags[i]
        xis = []
        rarr = []
        n_converged = 0

        rarr, xis = utils.load_data(cat_tag, cf_tag, Nrealizations=Nrealizations, return_amps=False)
        #rarr.append(r_avg)
        #xis.append(xi)
        r_avg = rarr[0]
        rs.append(r_avg)
        r_arrs.append(rarr)
        cf_arrs.append(xis)     

        if show_bases:
            if 'tophat' in cf_tag:
                rmin = min(r_avg)
                rmax = max(r_avg)
                binwidth = float(cf_tag.split('bw')[-1])
                r_edges = np.arange(rmin, rmax+binwidth, binwidth)
                base = np.zeros((len(r_avg), len(r_edges))) #r_edges - 1 (bc edges not bins), +1 (bc ravg)
                base[:,0] = r_avg
                for rr in range(len(r_edges)-1):
                    base[:,rr+1] = [1.0 if r_edges[rr]<=r_avg[jj]<r_edges[rr+1] else 0.0 for jj in range(len(r_avg))]
            else:
                if 'baoiter' in cf_tag:
                    rmin, rmax = 36.0, 200.0
                    redshift = 0.57
                    bias = 2.0
                    #projfn = f'../tables/bases{cf_tag}_r{rmin}-{rmax}_z{redshift}_bias{bias}.dat'
                    projfn = f"../tables/bases{cat_tag}{cf_tag}_r{rmin}-{rmax}_z{redshift}_bias{bias}.dat"
                else:
                    projfn = f'../tables/bases{cf_tag}.dat'
                base = np.loadtxt(projfn)
                bmax = max(np.array([base[bb,1:] for bb in range(base.shape[0]) if xlim[0]<base[bb,0]<xlim[1]]).flatten())
                base[:,1:] /= bmax   
            bases.append(base)        
        
    true_fn = '{}/inputs/cat{}_Rh_xi.txt'.format(cat_dir, cat_tag)
    r_true, xi_true = np.loadtxt(true_fn, unpack=True)
    xi_true *= b1**2

    ax = plot_cf_err(rs, cf_arrs, r_true, xi_true, labels, colors, err=err, xlim=xlim, 
                errlim=errlim, cont=cont, bases=bases)
    
    if peak_fit:
        r_peak_guess = 100.0
        r_peak_arr, *_ = find_peaks_center(r_arrs, cf_arrs, r_peak_guess, bws=bws, r_widths=r_widths)
        for i in range(len(cf_tags)):
            r_med = np.nanmedian(r_peak_arr[i])
            r_p16 = np.nanpercentile(r_peak_arr[i], 16)
            r_p84 = np.nanpercentile(r_peak_arr[i], 84)
            ax[0].errorbar(r_med, 0.008+i*0.001, xerr=[[r_med-r_p16], [r_p84-r_med]], fmt='o', 
                           color=colors[i], markersize=4)
        if r_max_true is not None:
            ax[0].errorbar(r_max_true, 0.008-0.001, fmt='o', color='k', markersize=4)
        return ax, r_peak_arr
    else:
        return ax


def plot_bases(bases, colors, names=None, rescale_by=None):
    plt.figure(figsize=(7,5))
    bases.shape
    r = bases[:,0]
    nbases = len(bases[0])-1
    if names is None:
        names = [None]*nbases
    for i in range(nbases):
        #norm = np.mean(bases[:,i])
        base = bases[:,i+1]
        if rescale_by is not None:
            base *= rescale_by[i]
        plt.plot(r, base, color=base_colors[i], label='{}'.format(names[i]), lw=2)
    if names is not None:
        plt.legend()

    plt.xlabel(r'r (h$^{-1}$ Mpc)')
    plt.ylabel('bases')
