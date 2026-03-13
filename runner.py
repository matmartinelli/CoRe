import sys,os
import numpy as np
import pandas as pd

from bios import read
from copy import deepcopy
from time import time

#MMmod: TODO
#Polynomial (and other methods) to be added and tested
from GaussianProcess.engine import GPCalculator

from Reconstruction.reconstructor import DerivedFunction

info = read(sys.argv[1])

#MMmod: TODO 
#Create output folder if not present

#Loop over the different reconstructions
for recon_key,recon_dict in info['reconstructions'].items():

    print('')
    print('STARTING RECONSTRUCTION OF {} OBSERVABLE'.format(recon_key))
    
    #MMmod: TODO
    #Add here settings test to stop code if needed

    #1) Reading data
    recon_dict['dataset'] = pd.read_csv(recon_dict['data_path'],sep='\s+',header=0)
    recon_dict['covmat']  = pd.read_csv(recon_dict['covmat_path'],sep='\s+',header=0)
    if 'fiducial_path' in recon_dict and recon_dict['fiducial_path'] != None:
        recon_dict['fiducial'] = pd.read_csv(recon_dict['fiducial_path'],sep='\s+',header=0)


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
        print('GP done in {:.2f} s'.format(time()-tini))

        #Reconstruction diagnostic
        #MMmod: TODO
        #Here we call the GP scorer

    else:
        sys.exit('ONLY GP AVAILABLE FOR NOW')

    #3) Saving results to file
    recon_dict['recon_means'].to_csv(info['outroot']+'_'+recon_key+'_means.txt',sep='\t',header=0)
    recon_dict['recon_covmat'].to_csv(info['outroot']+'_'+recon_key+'_covmat.txt',sep='\t',header=0)
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
