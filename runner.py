import sys,os
import numpy as np
import pandas as pd

from bios import read
from copy import deepcopy
from time import time

from scipy.interpolate import interp1d

#MMmod: TODO
#Polynomial (and other methods) to be added and tested
from GaussianProcess.engine import GPCalculator

#MMmod: TODO
#Generalize scorer and move it out of GP
from tools.diagnostics import Scorer

from Reconstruction.reconstructor import DerivedFunction

info = read(sys.argv[1])

#MMmod: TODO 
#Create output folder if not present

#Loop over the different reconstructions
fiducial = {}
for recon_key,recon_dict in info['reconstructions'].items():

    print('')
    print('STARTING RECONSTRUCTION OF {} OBSERVABLE'.format(recon_key))
    
    #MMmod: TODO
    #Add here settings test to stop code if needed

    #1) Reading data
    recon_dict['dataset'] = pd.read_csv(recon_dict['data_path'],sep='\s+',header=0)
    recon_dict['covmat']  = pd.read_csv(recon_dict['covmat_path'],sep='\s+',header=0)
    if 'fiducial_path' in recon_dict and recon_dict['fiducial_path'] != None:
        fidtable = pd.read_csv(recon_dict['fiducial_path'],sep='\s+',header=0)
        fiducial[recon_key] = {}
        for col in fidtable.columns:
            if col != 'x':
                fiducial[recon_key][col] = interp1d(fidtable['x'],fidtable[col])



    #2) Perform reconstruction
    if recon_dict['recon_method']['name'] == 'GaussianProcess':
        gp = GPCalculator(recon_dict['dataset'],recon_dict['covmat'],kernel_type=recon_dict['recon_method']['kernel'])

        N    = recon_dict['recon_method']['N']
        xmin = recon_dict['recon_method']['xmin']
        xmax = recon_dict['recon_method']['xmax']
        x_recon = np.linspace(xmin,xmax,N)
        kwargs = {'method': recon_dict['recon_method']['pars_selection']}
        if 'n_samples' in recon_dict['recon_method']:
            kwargs['n_samples'] = recon_dict['recon_method']['n_samples']
        else:
            kwargs['n_samples'] = 100

        tini = time()
        means, joint_cov, lml, gpinfo = gp.reconstruct(x_recon,**kwargs)
        recon_dict['recon_means']  = means
        recon_dict['recon_covmat'] = joint_cov

        if info['chatty']:
            print('GP done in {:.2f} s'.format(time()-tini))


    else:
        sys.exit('ONLY GP AVAILABLE FOR NOW')

    scorer = Scorer(recon_dict['recon_means'],recon_dict['recon_covmat'],chatty=info['chatty'])

    if recon_key in fiducial:
        theory_df = pd.DataFrame({'x': x_recon}|{func: interp(x_recon) for func,interp in fiducial[recon_key].items()})
        res         = scorer.score_against_theory(theory_df)

        #df['Mahalanobis score'] = res['red_chi2']

        res = scorer.score_pointwise(theory_df)

        #df['Diagonal score'] = res['red_chi2']

    datasets = [{'type': 'f',
                 'df': recon_dict['dataset'],
                 'cov': recon_dict['covmat']}]

    res = scorer.score_against_data(datasets)

    #df['Data score'] = res['red_chi2']

    #3) Saving results to file
    recon_dict['recon_means'].to_csv(info['outroot']+'_'+recon_key+'_means.txt',sep='\t',header=True,index=False)
    recon_dict['recon_covmat'].to_csv(info['outroot']+'_'+recon_key+'_covmat.txt',sep='\t',header=True,index=False)
    #TODO save diagnostic and settings here


#4) Create dictionaries for derived functions
all_recons = {k: v['recon_means'] for k,v in info['reconstructions'].items()}
all_covmats = {k: v['recon_covmat'] for k,v in info['reconstructions'].items()}

#Loop over the different derived functions

for func_key,func_sets in info['derived_functions'].items():

    print('')
    print('DERIVING {} FUNCTION'.format(func_key))

    tini = time()
    reconstructor = DerivedFunction(all_recons,all_covmats,sampler=func_sets['sampler'],run_options=func_sets['run_options'],chatty=info['chatty'])

    derived_logic = eval(func_sets['logic']) 

    sample = reconstructor.run(derived_logic,func_key,sigma_width=5)

    print('Function derived in {:.2f}'.format(time()-tini))

    
    derived_mean = pd.DataFrame({'x': reconstructor.x_recon})
    indices = [sample.index[func_key+'_'+str(ind)] for ind in range(len(reconstructor.x_recon))]

    derived_mean['value'] = sample.getMeans()[indices]
    ##MMmod: TODO
    #Do we want this error or the full covmat?
    derived_mean['error'] = np.sqrt(sample.getVars()[indices])

    derived_mean.to_csv(info['outroot']+'_derived_{}_function.txt'.format(func_key),sep='\t',header=True,index=False)

    #MMmod: TODO
    #Run diagnostic on derived
