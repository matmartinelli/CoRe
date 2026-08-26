import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

from gui_utils.plots import plot_data, plot_observable_recon
from gui_utils.sidebar_step1 import render_sidebar_step1
from gui_utils.sidebar_step2 import render_sidebar_step2
from gui_utils.analysis import st_capture_output, run_single_reconstruction

st.set_page_config(page_title="CoRe Reconstruction Pipeline", layout="wide")

if "step" not in st.session_state:
    st.session_state.step = 1
if "data_store" not in st.session_state:
    st.session_state.data_store = {}
if "recon_configs" not in st.session_state:
    st.session_state.recon_configs = []
if "outroot" not in st.session_state:
    st.session_state.outroot = ""
if "eigen_trunc_factor" not in st.session_state:
    st.session_state.eigen_trunc_factor = 1e-12

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
                    plot_data(data_df, name, x_label, y_labels)

elif st.session_state.step == 2:
    render_sidebar_step2()

    st.title("CoRe Reconstruction Pipeline")
    st.subheader("Step 2: Method Reconstruction & Comparison")

    if not st.session_state.recon_configs:
        st.warning("⚠️ Add one or more reconstruction methods in the sidebar to run the reconstruction pipeline.")
    else:
        if st.button("🚀 Run All Reconstructions & Compare", type="primary"):
            num_datasets = len(st.session_state.data_store)
            outroot = st.session_state.get("outroot", "")
            factor = st.session_state.get("eigen_trunc_factor", 1e-12)

            for name, config in st.session_state.data_store.items():
                data_df  = config["data_df"]
                cov_df   = config["cov_df"]
                x_label  = config["x_label"]
                y_labels = config["y_labels"]

                st.markdown(f"### Comparison Results: `{name}`")
                
                with st.expander("📺 Live Computation Logs", expanded=True):
                    log_placeholder = st.empty()
                
                recon_dicts = []

                with st_capture_output(log_placeholder):
                    for cfg in st.session_state.recon_configs:
                        recon_res = run_single_reconstruction(
                            cfg=cfg, 
                            data_df=data_df, 
                            cov_df=cov_df, 
                            y_labels=y_labels, 
                            dataset_name=name,
                            outroot=outroot,
                            num_datasets=num_datasets,
                            eigen_trunc_factor=factor
                        )
                        recon_dicts.append(recon_res)

                # Layout: Plot on left, Styled Diagnostic Table on right
                col_plot, col_diag = st.columns([1.1, 1])

                with col_plot:
                    st.markdown("**Reconstruction Plot**")
                    fig_res = plot_observable_recon(data_df, name, recon_dicts, x_label, y_labels)

                with col_diag:
                    st.markdown("**Diagnostic Scores**")
                    formatted_scores = {}
                    p_trunc_values = {}

                    for recon in recon_dicts:
                        label = recon['label']
                        s = recon.get('score', {})

                        dof = s.get('dof', 0)
                        trunc_dof = s.get('truncated_dof', 0)
                        n_cut = dof - trunc_dof
                        p_trunc = s.get('truncated_p_value', 1.0)

                        p_trunc_values[label] = p_trunc

                        formatted_scores[label] = {
                            "Total chi2": s.get('red_chi2'),
                            "p-value": s.get('p_value'),
                            "Truncated modes": n_cut,
                            "Truncated chi2": s.get('truncated_red_chi2'),
                            "Truncated p-value": p_trunc
                        }

                    if formatted_scores:
                        scores_df = pd.DataFrame(formatted_scores)

                        def highlight_cols(col):
                            p_val = p_trunc_values.get(col.name, 1.0)
                            if p_val <= 0.01:
                                color = 'background-color: #f8d7da; color: #721c24;'  # Red
                            elif p_val <= 0.05:
                                color = 'background-color: #fff3cd; color: #856404;'  # Yellow
                            else:
                                color = 'background-color: #d4edda; color: #155724;'  # Green
                            return [color] * len(col)

                        styled_df = scores_df.style.apply(highlight_cols, axis=0).format(
                            lambda x: f"{int(x)}" if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) else (f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x))
                        )

                        st.dataframe(styled_df, use_container_width=True)

            st.success("Reconstruction complete!")
