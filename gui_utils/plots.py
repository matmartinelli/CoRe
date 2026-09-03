import numpy as np
import pandas as pd
import seaborn as sb
import streamlit as st

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import getdist.plots as gplots

rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

red    = '#8e001c'
yellow = '#ffb302'

sidelegend = {
    'bbox_to_anchor': (1.04, 0.5),
    'loc': "center left",
    'frameon': False
}
bottomlegend = {
    'bbox_to_anchor': (0.35, -0.2),
    'loc': "center left",
    'frameon': False,
    'ncols': 3
}


def clean_tex_string(s: str) -> str:
    """Strips surrounding dollar signs and whitespace from LaTeX input strings."""
    s = str(s).strip()
    if s.startswith('$') and s.endswith('$') and len(s) > 1:
        return s[1:-1].strip()
    return s


def ensure_math_mode(s: str) -> str:
    """Ensures a string is cleanly wrapped in LaTeX math mode without double dollar signs."""
    clean = clean_tex_string(s)
    if not clean:
        return ""
    return f"${clean}$"


def plot_data(data_df, name, x_label, y_labels):
    fig, ax = plt.subplots(nrows=len(y_labels), ncols=1, figsize=(6, 4 * len(y_labels)))
    axes = ax if len(y_labels) > 1 else [ax]

    for ind, ylabel in enumerate(y_labels):
        fnum = str(ind + 1) if len(y_labels) > 1 else ''
        axes[ind].errorbar(
            data_df['x'], 
            data_df['f' + fnum], 
            yerr=data_df['f' + fnum + '_err'], 
            fmt='o', 
            label='Data', 
            color='black'
        )
        axes[ind].set_xlabel(ensure_math_mode(x_label))
        axes[ind].set_ylabel(ensure_math_mode(ylabel))
        axes[ind].grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc='best', frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_covmat(cov_df, Ndata, x_label, y_labels):
    cov = cov_df.copy()

    columns = {}
    for i, ylab in enumerate(y_labels):
        columns = columns | {f'f{i+1}_{ind}': ensure_math_mode(f'{ylab}_{{{ind}}}') for ind in range(Ndata)}

    cov = cov.rename(columns=columns, index=columns)

    fig, ax = plt.subplots(figsize=(6, 4))
    sb.heatmap(cov, ax=ax, xticklabels=False, yticklabels=False)
    st.pyplot(fig)
    plt.close(fig)


def plot_observable_recon(dataset, data_name, recon_dicts, x_label, y_labels):
    colors = [red, yellow, 'purple', 'cyan']
    Nfuncs = len([col for col in dataset.columns if col != 'x' and '_err' not in col])

    fig, ax = plt.subplots(ncols=Nfuncs, nrows=4, sharex=True, figsize=(5 * Nfuncs, 12))
    axes = ax if Nfuncs > 1 else ax.reshape(4, 1)

    for j in range(Nfuncs):
        fnum = str(j + 1) if Nfuncs > 1 else ''
        axes[0, j].errorbar(
            dataset['x'], 
            dataset['f' + fnum], 
            yerr=dataset['f' + fnum + '_err'], 
            fmt='o', 
            label='Data', 
            color='black'
        )

    clean_x = clean_tex_string(x_label)

    for ind, recon in enumerate(recon_dicts):
        means = recon['means']
        col_color = colors[ind % len(colors)]

        for j in range(Nfuncs):
            fnum = str(j + 1) if Nfuncs > 1 else ''
            clean_y = clean_tex_string(y_labels[j]) if j < len(y_labels) else f"f_{{{j+1}}}"

            derlabels = [
                f"${clean_y}$",
                f"$d\\,{clean_y}/d\\,{clean_x}$",
                f"$d^2\\,{clean_y}/d\\,{clean_x}^2$",
                f"$\\int_0^z {clean_y}\\,d{clean_x}$"
            ]

            i = 0
            for deriv, derlabel in zip(['f', 'd1', 'd2', 'int'], derlabels):
                if deriv + fnum in means.columns:
                    if recon['method'] == 'Gaussian Process':
                        axes[i, j].fill_between(
                            means['x'], 
                            means[deriv + fnum] + means[deriv + fnum + '_err'],
                            means[deriv + fnum] - means[deriv + fnum + '_err'], 
                            alpha=0.2, 
                            color=col_color
                        )
                        axes[i, j].plot(
                            means['x'], 
                            means[deriv + fnum], 
                            color=col_color, 
                            label=recon['label']
                        )
                    elif recon['method'] == 'Binned':
                        axes[i, j].errorbar(
                            means['x'], 
                            means[deriv + fnum], 
                            yerr=means[deriv + fnum + '_err'], 
                            ls='', 
                            color=col_color, 
                            alpha=1,
                            fmt='o', 
                            ms=4, 
                            capsize=3, 
                            elinewidth=2.0, 
                            label=recon['label']
                        )

                    axes[i, j].grid(True, linestyle="--", alpha=0.6)

                axes[i, j].set_ylabel(derlabel)
                i += 1

    for i in range(Nfuncs):
        axes[-1, i].set_xlabel(f"${clean_x}$")

    fig.align_ylabels()
    plt.suptitle(f"Reconstruction: {data_name}", y=1.)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=len(labels), frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    st.pyplot(fig)
    plt.close(fig)


def plot_derived_triangle(derived_res, derived_names, x_recon, color, label):
    """Generates GetDist triangle plot for derived reconstruction functions."""
    g = gplots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)

    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.legend_fontsize = 20
    g.settings.axis_marker_color = 'black'
    g.settings.axis_marker_ls = '--'
    g.settings.axis_marker_lw = 1
    g.settings.axis_tick_x_rotation = 45

    param_names = []
    for d_name in derived_names:
        param_names.extend([f"{d_name}_{i}" for i in range(len(x_recon))])

    g.triangle_plot(
        [derived_res],
        param_names,
        filled=True,
        contour_lws=2,
        contour_colors=[color],
        legend_labels=[label]
    )
    g.fig.align_ylabels()
    g.fig.align_xlabels()
    return g.fig


def plot_derived_summary(all_derived_results, recon_configs, x_label, derived_name, tex_label=""):
    """Plots combined 1D summary of derived reconstructions using custom TeX labels."""
    fig = plt.figure()
    colors = [red, yellow, 'purple', 'black']

    for ind, (recon_label, derived_res) in enumerate(all_derived_results.items()):
        sample = derived_res['sample']
        x_recon = derived_res['x_recon']
        if sample is None:
            continue

        all_pars = sample.getParamNames().list()
        means = {par: val for par, val in zip(all_pars, sample.getMeans())}
        errors = {par: np.sqrt(val) for par, val in zip(all_pars, sample.getVars())}

        derpars = [f"{derived_name}_{i}" for i in range(len(x_recon))]

        reconstruction = pd.DataFrame({'x': x_recon})
        reconstruction['value'] = [val for par, val in means.items() if par in derpars]
        reconstruction['error'] = [val for par, val in errors.items() if par in derpars]

        color = colors[ind % len(colors)]
        method = recon_configs[ind]["method"] if ind < len(recon_configs) else 'Gaussian Process'

        if method == 'Gaussian Process':
            plt.fill_between(
                reconstruction['x'],
                reconstruction['value'] + reconstruction['error'],
                reconstruction['value'] - reconstruction['error'],
                alpha=0.2,
                color=color
            )
            plt.plot(
                reconstruction['x'],
                reconstruction['value'],
                lw=2,
                color=color,
                label=recon_label
            )
        elif method == 'Binned':
            plt.errorbar(
                reconstruction['x'],
                reconstruction['value'],
                yerr=reconstruction['error'],
                ls='',
                color=color,
                alpha=1,
                fmt='o',
                ms=4,
                capsize=3,
                elinewidth=2.0,
                label=recon_label
            )

    plt.xlabel(ensure_math_mode(x_label))
    ylabel_str = tex_label if tex_label else derived_name
    plt.ylabel(ensure_math_mode(ylabel_str))
    plt.legend(loc='best')
    return fig
