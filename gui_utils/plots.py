import numpy as np
import pandas as pd

import streamlit as st

#Plotting
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

red    = '#8e001c'
yellow = '#ffb302'

sidelegend = {'bbox_to_anchor': (1.04,0.5),
              'loc': "center left",
              'frameon': False}
bottomlegend = {'bbox_to_anchor': (0.35,-0.2),
                'loc': "center left",
                'frameon': False,
                'ncols': 3}

def plot_data(data_df,name,x_label,y_labels):

    fig, ax = plt.subplots(nrows=1,ncols=len(y_labels),figsize=(6, 4))
    if len(y_labels) > 1:
        axes = ax
    else:
        axes = [ax]
    for ind,ylabel in enumerate(y_labels):
        if len(y_labels)>1:
            fnum = str(ind+1)
        else:
            fnum = ''
        axes[ind].errorbar(data_df['x'], data_df['f'+fnum], yerr=data_df['f'+fnum+'_err'], fmt='o', label='Data',color='black')
        axes[ind].set_xlabel(x_label)
        axes[ind].set_ylabel(y_labels[ind])
        axes[ind].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc='best',frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return None


def plot_observable_recon(dataset,data_name,recon_dicts,x_label,y_labels):

    colors = [red,yellow,'purple','cyan']
    Nfuncs  = len([col for col in dataset.columns if col != 'x' and '_err' not in col])

    fig, ax = plt.subplots(ncols=Nfuncs,nrows=4,sharex=True,figsize=(5*Nfuncs,12))
    if Nfuncs > 1:
        axes = ax
    else:
        axes = ax.reshape(4,1)

    for j in range(Nfuncs):
        if Nfuncs > 1:
            fnum = str(j+1)
        else:
            fnum = ''
        axes[0,j].errorbar(dataset['x'], dataset['f'+fnum], yerr=dataset['f'+fnum+'_err'], fmt='o', label='Data', color='black')

    for ind,recon in enumerate(recon_dicts):
        means = recon['means']

        for j in range(Nfuncs):
            if Nfuncs > 1:
                fnum = str(j+1)
            else:
                fnum = ''
            i = 0
            for deriv,derlabel in zip(['f','d1','d2','int'],[r'${}$'.format(y_labels[j]),r'$d\,{}/d\,{}$'.format(y_labels[j],x_label),
                                                             r'$d^2\,{}/d\,{}^2$'.format(y_labels[j],x_label),r'$\int_0^z{}$'.format(y_labels[j])]):
                if deriv+fnum in means.columns:
                    if recon['method'] == 'Gaussian Process':
                        axes[i,j].fill_between(means['x'],means[deriv+fnum]+means[deriv+fnum+'_err'],
                                               means[deriv+fnum]-means[deriv+fnum+'_err'],alpha=0.2,color=colors[ind])
                        axes[i,j].plot(means['x'],means[deriv+fnum],color=colors[ind],label=recon['label'])
                    elif recon['method'] == 'Binned':
                        axes[i,j].errorbar(means['x'],means[deriv+fnum],yerr=means[deriv+fnum+'_err'],ls='',color=colors[ind],alpha=1,
                                           fmt='o',ms=4,capsize=3,elinewidth=2.0,label=recon['label'])

                    axes[i,j].grid(True, linestyle="--", alpha=0.6)

                axes[i,j].set_ylabel(derlabel)
                i += 1

    for i in range(Nfuncs):
        axes[-1,i].set_xlabel(x_label)
    fig.align_ylabels()
    plt.suptitle(f"Reconstruction: {data_name}",fontweight='bold',y=1.)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(0.5,0.92),ncol=len(labels),frameon=False)
    fig.subplots_adjust(top=0.82)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return None
