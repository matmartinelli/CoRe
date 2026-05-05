import sys,os
import numpy as np
import pandas as pd

from bios import read
from copy import deepcopy
from time import time

from scipy.interpolate import interp1d

#Reconstructions tools
from reconstruction_tools.GaussianProcess import GPCalculator

from utils.diagnostics import Scorer

#Derived function reconstructor
from derived_function.reconstructor import DerivedFunction

info = read(sys.argv[1])

#MMmod: TODO 
#Create output folder if not present

#Loop over the different reconstructions
fiducial = {}
for data_key,dataset in info['datasets'].items():

    print('')
    print('STARTING RECONSTRUCTION OF {} OBSERVABLE'.format(data_key))
    
    #MMmod: TODO
    #Add here settings test to stop code if needed

    #1) Reading data
    dataset['dataset'] = pd.read_csv(dataset['data_path'],sep='\s+',header=0)
    dataset['covmat']  = pd.read_csv(dataset['covmat_path'],sep='\s+',header=0)
    if 'fiducial_path' in dataset and dataset['fiducial_path'] != None:
        fidtable = pd.read_csv(dataset['fiducial_path'],sep='\s+',header=0)
        fiducial[data_key] = {}
        for col in fidtable.columns:
            if col != 'x':
                fiducial[data_key][col] = interp1d(fidtable['x'],fidtable[col])



    #2) Perform reconstruction
    if info['reconstruction_settings']['name'] == 'GaussianProcess':

        gp = GPCalculator(dataset['dataset'],dataset['covmat'],kernel_type=info['reconstruction_settings']['kernel'],chatty=info['chatty'])

        N    = info['reconstruction_settings']['N']
        xmin = info['reconstruction_settings']['xmin']
        xmax = info['reconstruction_settings']['xmax']
        x_recon = np.linspace(xmin,xmax,N)
        kwargs = {'method': info['reconstruction_settings']['pars_selection']}
        if 'n_samples' in info['reconstruction_settings']:
            kwargs['n_samples'] = info['reconstruction_settings']['n_samples']
        else:
            kwargs['n_samples'] = 100

        tini = time()
        means, joint_cov, lml, gpinfo = gp.reconstruct(x_recon,**kwargs)
        dataset['recon_means']  = means
        dataset['recon_covmat'] = joint_cov
        dataset['recon_info']   = gpinfo

        if info['chatty']:
            print('GP done in {:.2f} s'.format(time()-tini))


    else:
        sys.exit('ONLY GP AVAILABLE FOR NOW')

    if info['want_score']: 
        scorer = Scorer(dataset['recon_means'],dataset['recon_covmat'],chatty=info['chatty'])

        if data_key in fiducial:
            theory_df = pd.DataFrame({'x': x_recon}|{func: interp(x_recon) for func,interp in fiducial[data_key].items()})
            res         = scorer.score_against_theory(theory_df)

            #df['Mahalanobis score'] = res['red_chi2']
 
            res = scorer.score_pointwise(theory_df)

            #df['Diagonal score'] = res['red_chi2']

        datasets = [{'type': 'f',
                     'df': dataset['dataset'],
                     'cov': dataset['covmat']}]

        res = scorer.score_against_data(datasets)

    #df['Data score'] = res['red_chi2']

    #3) Saving results to file
    dataset['recon_means'].to_csv(info['outroot']+'_'+data_key+'_means.txt',sep='\t',header=True,index=False)
    dataset['recon_covmat'].to_csv(info['outroot']+'_'+data_key+'_covmat.txt',sep='\t',header=True,index=False)
    np.save(info['outroot']+'_'+data_key+'_info.npy',dataset['recon_info'])
    #TODO save diagnostic and settings here


#4) Create dictionaries for derived functions
all_recons = {k: v['recon_means'] for k,v in info['datasets'].items()}
all_covmats = {k: v['recon_covmat'] for k,v in info['datasets'].items()}

#Loop over the different derived functions (if present)

if 'derived_functions' in info:
    for func_key,func_sets in info['derived_functions'].items():

        print('')
        print('DERIVING {} FUNCTION'.format(func_key))

        tini = time()
        reconstructor = DerivedFunction(all_recons,all_covmats,func_sets['method_dict'],chatty=info['chatty'])

        derived_logic = eval(func_sets['logic']) 
        
        if func_sets['method_dict']['type'] == 'sampling':
            sample = reconstructor.run(derived_logic,func_key,sigma_width=5)
        elif func_sets['method_dict']['type'] == 'realizations':
            sample = reconstructor.run(derived_logic,func_key)
        else:
            sys.exit('UNKNOWN DERIVED FUNCTION METHOD: {}'.format(func_sets['method_dict']['type']))

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
