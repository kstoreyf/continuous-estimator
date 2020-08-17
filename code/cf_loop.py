import cf


def main():

    L = 750
    cat_tags = [f'_L{L}_n5e-5_z057', f'_L{L}_n1e-4_z057', f'_L{L}_n2e-4_z057']
    #cf_tags = ['_tophat_bw12', '_spline3_bw6']
    cf_tags = ['_tophat_bw3', '_tophat_bw12']

    for cat_tag in cat_tags:
        for cf_tag in cf_tags:
            tagparts = cf_tag.split('_')
            print(tagparts)
            proj = tagparts[1]
            binwidth = int(tagparts[2][len('bw'):])   
            kwargs = {}
            if proj.startswith('spline'):
                kwargs['order'] = int(proj[len('spline'):])
                proj = 'spline'

            print(L, cat_tag, proj, cf_tag, binwidth, kwargs)
            cf.compute_xis(L, cat_tag, proj, cf_tag, binwidth=binwidth, kwargs=kwargs, Nrealizations=1000, nthreads=24)


if __name__=="__main__":
    main()
