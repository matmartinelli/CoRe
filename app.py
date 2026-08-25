import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from CoRe.reconstruction_tools.Binning         import BinnedCalculator
from CoRe.reconstruction_tools.GaussianProcess import GPCalculator
from CoRe.utils.diagnostics                    import Scorer

from gui_utils.plots         import plot_data,plot_observable_recon
from gui_utils.sidebar_step1 import render_sidebar_step1
from gui_utils.sidebar_step2 import render_sidebar_step2

epsilon = 1.e-12

st.set_page_config(page_title="CoRe Reconstruction Pipeline", layout="wide")

if "step" not in st.session_state:
    st.session_state.step = 1
if "data_store" not in st.session_state:
    st.session_state.data_store = {}
if "recon_configs" not in st.session_state:
    st.session_state.recon_configs = []

# PAGE ROUTING
if st.session_state.step == 1:
    render_sidebar_step1()

    st.title("CoRe Reconstruction Pipeline")
    st.subheader("Step 1: Upload & Inspect Data")

    if not st.session_state.data_store:
        st.info("👈 Please start by uploading at least one dataset and covariance pair using the sidebar.")
    else:
        for name, config in st.session_state.data_store.items():
            data_df = config["data_df"]
            x_label = config["x_label"]
            y_labels = config["y_labels"]

            with st.expander(f"📊 Dataset Preview: `{name}`", expanded=True):
                col_tbl, col_plt = st.columns([1, 1])
                with col_tbl:
                    st.markdown("**Data Table:**")
                    st.dataframe(data_df, use_container_width=True)

                with col_plt:
                    st.markdown("**Raw Data Plot:**")
                    plot_data(data_df,name,x_label,y_labels)

elif st.session_state.step == 2:
    N, xmin, xmax = render_sidebar_step2()

    st.title("CoRe Reconstruction Pipeline")
    st.subheader("Step 2: Method Reconstruction & Comparison")

    if not st.session_state.recon_configs:
        st.warning("👈 Add one or more reconstruction methods in the sidebar to run the reconstruction pipeline.")
    else:
        if st.button("🚀 Run All Reconstructions & Compare", type="primary"):
            x_recon = np.linspace(xmin, xmax, N)

            for name, config in st.session_state.data_store.items():
                data_df  = config["data_df"]
                cov_df   = config["cov_df"]
                x_label  = config["x_label"]
                y_labels = config["y_labels"]

                st.markdown(f"### Comparison Results: `{name}`")
                recon_dicts = []

                with st.spinner(f"Computing reconstructions for '{name}'..."):
                    for cfg in st.session_state.recon_configs:
                        m_type = cfg["method"]

                        if m_type == 'Binned':
                            bin_engine = BinnedCalculator(data_df, cov_df, method=cfg["binning_method"])
                            fid_func = cfg.get("fiducial_func", None)
                            means, cov, mask = bin_engine.reconstruct(x_recon, fiducial=fid_func)
                            joint_cov = pd.DataFrame(cov.values + np.eye(len(cov)) * epsilon, columns=cov.columns, index=cov.index)

                        elif m_type == 'Gaussian Process':
                            kwargs = {'n_samples': cfg["n_samples"]}
                            if cfg["hp_method"] == 'BAYESIAN':
                                kwargs.update({'sampler_name': cfg["sampler"], 'sampler_options': cfg["sampsets"]})

                            gp = GPCalculator(data_df, cov_df, kernel_type=cfg["kernel"], chatty=False)
                            means, joint_cov, lml, info = gp.reconstruct(x_recon, method=cfg["hp_method"], **kwargs)

                        recon_dicts.append({
                            'means': means,
                            'covmat': joint_cov,
                            'method': m_type,
                            'label': cfg['label']
                        })

                fig_res = plot_observable_recon(data_df, name, recon_dicts, x_label, y_labels)
                if fig_res is not None:
                    st.pyplot(fig_res)
                    plt.close(fig_res)

            st.success("Reconstruction complete!")
