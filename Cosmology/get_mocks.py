import sys,os
import numpy as np
import pandas as pd

from scipy.stats       import rv_continuous
from scipy.integrate   import trapezoid
from scipy.interpolate import interp1d

clight = 299792.458

class MockMaker:

    def __init__(self,theory,mock_data_path=None):

        self.theory = theory

        self.mock_data_path = mock_data_path

    def get_theory(self):

        return results

    def BAO_mock(self,survey='SKAO',correlation=None):
   
        bin_edges = np.linspace(0.2,2.0,18)
        z_BAO     = (bin_edges[:-1] + bin_edges[1:]) / 2
   
        DH = lambda x: (clight/self.theory['Hubble'](x))/self.theory['rd']
        DM = lambda x: self.theory['DM'](x)/self.theory['rd']

        if survey == 'SKAO':
            bin_edges = np.linspace(0.2,2.0,18)
            z_BAO     = (bin_edges[:-1] + bin_edges[1:]) / 2
   
            DH_error = DH(z_BAO)*np.array([1.8,1.17,0.99,0.79,0.7,0.64,0.61,0.57,0.67,0.69,0.8,0.95,1.16,1.51,2.12,3,5.3])*1.e-2
            DM_error = DM(z_BAO)*np.array([1.1,0.76,0.59,0.5,0.44,0.42,0.4,0.38,0.42,0.45,0.54,0.63,0.83,1.12,1.55,2.20,3.95])*1.e-2
        else:
            sys.exit('Unknown survey: {}'.format(survey))
   
        DH_noisy = np.random.normal(DH(z_BAO), DH_error)
        DM_noisy = np.random.normal(DM(z_BAO), DM_error)
   
        #TODO: modelling covariance
        if correlation == None:
            print('WARNING! Assuming no correlation in BAO data')
            correlation = np.zeros(len(z_BAO))
  
        data_df = pd.DataFrame({'x' : z_BAO,
                                'f1': DM_noisy,
                                'f1_err': DM_error,
                                'f2': DH_noisy,
                                'f2_err': DH_error,
                                'r_MH': correlation})

        covmat_1 = pd.DataFrame(np.diag(DM_error**2),columns=['f1_{}'+format(ind) for ind in range(len(data_df['x']))])
        covmat_1.index = covmat_1.columns

        covmat_2 = pd.DataFrame(np.diag(DH_error**2),columns=['f2_{}'+format(ind) for ind in range(len(data_df['x']))])
        covmat_2.index = covmat_2.columns

        cov_cross = np.diag([row['f1_err']*row['f2_err']*row['r_MH'] for ind,row in data_df.iterrows()])

        data_df = data_df.drop(columns=['r_MH'])

        joint_matrix = np.block([[covmat_1.values,  cov_cross],
                                [cov_cross.T, covmat_2.values]])

        N = len(data_df)
        labels = [f"f1_{i}" for i in range(N)] + [f"f2_{i}" for i in range(N)]
        covmat_df = pd.DataFrame(joint_matrix, columns=labels, index=labels)
   
   
        if self.mock_data_path != None:
           data_df.to_csv(mock_data_path+'BAOmock_'+survey+'_dataset.txt',header=True,index=False,sep='\t')
           covmat_df.to_csv(mock_data_path+'BAOmock_'+survey+'_covmat.txt',header=True,index=True,sep='\t')

   
        return data_df,covmat_df

    def SN_mock(self,MB=-19.23,survey='LSST'): #MM: check!

        mB = lambda x: 5*np.log10(self.theory['dL_EM'](x))+25+MB

        if survey == 'LSST':
            N_SN = 8800
            zmin = 0.1
            zmax = 1.0

            dN_dz = lambda z: 1.53e-4 * ((1+z)/1.5)**2.14 * (self.theory['H0']/70)**3

            z    = np.linspace(zmin,zmax,N_SN)
            p_z  = dN_dz(z)
            p_z /= np.sum(p_z)
            z_SN = np.sort(np.random.choice(z,size=N_SN, p=p_z, replace = False))

            sigma_flux = 0.01
            sigma_scat = 0.025
            sigma_intr = 0.12

            mB_error   = np.array([np.sqrt((np.random.normal(loc=0.,scale=0.01)*z)**2+sigma_flux**2+sigma_scat**2.+sigma_intr**2.)
                                   for z in z_SN])

        mB_noisy = np.random.normal(mB(z_SN), mB_error)


        data_SN = {'x' : z_SN,
                   'f': mB_noisy,
                   'f_err': mB_error}

        #TODO: modelling covariance
        covmat_SN = np.zeros((len(mB_error), len(mB_error)))
        np.fill_diagonal(covmat_SN, mB_error ** 2)
        print('WARNING! Assuming diagonal covariance')


        #Creating dataframe to save to file
        data_df   = pd.DataFrame.from_dict(data_SN)
        covmat_df = pd.DataFrame(covmat_SN,columns=['f_{}'.format(i) for i in data_df.index])
        covmat_df.index = covmat_df.columns

        if self.mock_data_path != None:
            data_df.to_csv(mock_data_path+'SNmock_'+survey+'_data.txt',header=True,index=False,sep='\t')
            covmat_df.to_csv(mock_data_path+'SNmock_'+survey+'_covmat.txt',header=True,index=False,sep='\t')

        return data_df,covmat_df

    def BNS_merger_rate(self,z):
        if z <= 1:
            return 1 + 2 * z
        elif 1 < z < 5:
            return 3/4 * (5-z)
        else:
            return 0
            
    def get_events_redshifts(self,pz,zmin,zmax,Nsamp):
        
        class MyDist(rv_continuous):
            def _pdf(self, x):
                return pz(x)
                
        mydist = MyDist(a=zmin,b=zmax)
        zs = mydist.rvs(size=Nsamp)
        
        return zs

    def get_realistic_error_GW(self,z_GW,dL_GW,surveys):

        from GWFish.modules.detection    import Network,Detector
        from GWFish.modules.fishermatrix import compute_network_errors,compute_detector_fisher,compute_detector_fisher
        from GWFish.modules.waveforms    import IMRPhenomD, TaylorF2

        Ngw = len(z_GW)

        th_features = pd.DataFrame.from_dict({'z': z_GW,
                                              'luminosity_distance': dL_GW})

        #Generating random features

        #Sky location
        th_features['dec']  = np.arccos(np.random.uniform(low=-1, high=1,size=Ngw))
        th_features['ra']   = np.random.uniform(low=0, high=2*np.pi,size=Ngw)

        #Polarization
        th_features['psi'] = np.random.uniform(low=0, high=2*np.pi,size=Ngw)

        #Phase
        th_features['phase'] = np.random.uniform(low=0,high=2.*np.pi,size=Ngw)

        #System inclination ##MM: switch to arccos for 0-90?
        th_features['theta_jn'] = np.arccos(np.random.uniform(low=0, high=1,size=Ngw))

        #MM: check this!!!
        th_features['geocent_time'] = np.random.uniform(1735257618, 1766793618,size=Ngw)

        #MM: ASSUMING MONOCHROMATIC MASS
        th_features['mass_1'] = 1.4
        th_features['mass_2'] = 1.4
        Mtot = (th_features['mass_1']+th_features['mass_2'])
        eta  = th_features['mass_1']*th_features['mass_2']/(th_features['mass_1']+th_features['mass_2'])**2
        th_features['chirp_mass'] = (1+th_features['z'])*Mtot*eta**(3/5)
        th_features['mass_ratio'] = eta

        #MM: hard coded?
        freepars = ['theta_jn','luminosity_distance']
        SNR_cut  = 20
        detected, snr, errors, sky_localization = compute_network_errors(network = Network(detector_ids = surveys,
                                                                                           detection_SNR = (0., 0)),
                                                                                           parameter_values = th_features,
                                                                                           fisher_parameters=freepars,
                                                                                           waveform_model = 'IMRPhenomD',
                                                                                           save_matrices=True)


        for i,par in enumerate(freepars):
            th_features['err_'+par] = errors[:,i]

        th_features['SNR'] = snr
        observed = th_features[th_features['SNR']>=SNR_cut]

        return observed

    def GW_mock(self,z_min=0.001,z_max=10.,N_gw=20000,theta_cut=18,survey='ET'):

        N_gw      = 20000
        zmin      = 0.001
        zmax      = 10.
        if survey == 'ET':
            surveys   = ['ET']
        else:
            sys.exit('Unknown survey: {}'.format(survey))

        z_calc = np.linspace(zmin,zmax,N_gw)

        rate   = np.array([self.BNS_merger_rate(z) for z in z_calc])
        unnorm = rate*((4*np.pi*(self.theory['DM'](z_calc))**2)/
                       (self.theory['Hubble'](z_calc)*(1+z_calc)/clight))

        integral  = trapezoid(unnorm,x=z_calc)
        norm      = 1/integral
        norm_dist = norm*unnorm

        prob_z = interp1d(z_calc,norm_dist,kind='linear',bounds_error=False,fill_value=0)
        z_GW   = self.get_events_redshifts(prob_z,zmin,zmax,N_gw)
        dL_GW  = self.theory['dL_GW'](z_GW)

        print('')
        print('Distribution of events generated')

        observed    = self.get_realistic_error_GW(z_GW,dL_GW,surveys)
        z_GW        = observed['z']
        dL_GW       = observed['luminosity_distance']
        dL_GW_error = observed['err_luminosity_distance']
        theta_jn    = observed['theta_jn']

        dL_GW_noisy = np.random.normal(dL_GW,dL_GW_error)


        data_GW = {'z' : z_GW,
                   'dL': dL_GW_noisy,
                   'dL_err': dL_GW_error,
                   'theta_jn': np.rad2deg(theta_jn)}

        print('')
        print('Errors generated')

        data_df_uncut = pd.DataFrame.from_dict(data_GW)

        #MM: warrning! The visibility cut done in the paper is slightly more complicated
        #this is just an approximation
        data_df = data_df_uncut[data_df_uncut['theta_jn']<=theta_cut]

        data_df = data_df.drop(columns=['theta_jn'])

        data_df = data_df.rename(columns={'z': 'x','dL': 'f','dL_err': 'f_err'})

        data_df_uncut['Detection'] = ['Detected with EM counterpart' if val <= theta_cut else 'Detected'
                                      for val in data_df_uncut['theta_jn']]

        #TODO: model covariance
        covmat_GW = np.zeros((len(data_df['f_err']), len(data_df['f_err'])))
        np.fill_diagonal(covmat_GW,data_df['f_err'].values**2)

        covmat_df = pd.DataFrame(covmat_GW,columns=['f_{}'.format(i) for i in data_df.index])
        covmat_df.index = covmat_df.columns

        if self.mock_data_path != None:
            data_df.to_csv(mock_data_path+'GWmock_'+survey+'_data.txt',header=True,index=False,sep='\t')
            covmat_df.to_csv(mock_data_path+'GWmock_'+survey+'_covmat.txt',header=True,index=False,sep='\t')

        return data_df,covmat_df
