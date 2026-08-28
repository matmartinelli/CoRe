import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

from gui_utils.plots         import plot_data, plot_covmat, plot_observable_recon
from gui_utils.sidebar_step1 import render_sidebar_step1
from gui_utils.sidebar_step2 import render_sidebar_step2
from gui_utils.analysis      import st_capture_output, run_single_reconstruction

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


def format_val(val):
    if val is None:
        return "N/A"
    if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()):
        return f"{int(val)}"
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4g}"
    return str(val)


def get_status_badge(p_val):
    if p_val <= 0.01:
        return "🔴"
    elif p_val <= 0.05:
        return "🟡"
    else:
        return "🟢"


# PAGE ROUTING
if st.session_state.step == 1:
    render_sidebar_step1()

    st.title("CoRe Reconstruction Pipeline")
    st.subheader("Step 1: Upload & Inspect Data")

    if not st.session_state.data_store:
        st.info("👈 Please start by uploading at least one dataset and covariance pair using the sidebar.")
    else:
        for name, config in st.session_state.data_store.items():
            data_df  = config["data_df"]
            cov_df   = config["cov_df"] 
            x_label  = config["x_label"]
            y_labels = config["y_labels"]

            with st.expander(f"📊 Dataset Preview: `{name}`", expanded=True):
                col_tbl, col_plt = st.columns([1, 1])
                with col_tbl:
                    st.markdown("### Raw Data Plot:")
                    plot_data(data_df, name, x_label, y_labels)

                with col_plt:
                    st.markdown("### Data Covariance:")
                    Ndata = len(data_df.index)
                    plot_covmat(cov_df, Ndata, x_label, y_labels)

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

                # Layout: Plot on left, Markdown Diagnostic Table on right
                col_plot, col_diag = st.columns([1.1, 1])

                with col_plot:
                    st.markdown("### Reconstruction Plot")
                    fig_res = plot_observable_recon(data_df, name, recon_dicts, x_label, y_labels)

                with col_diag:
                    st.markdown("### Diagnostic Scores")

                    if recon_dicts:
                        # Construct Markdown Table Header
                        headers = ["Metric"] + [
                            f"{get_status_badge(r.get('score', {}).get('truncated_p_value', 1.0))} **{r['label']}**"
                            for r in recon_dicts
                        ]
                        
                        md_lines = ["| " + " | ".join(headers) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                        # Table Metrics Definitions
                        metrics_map = [
                            (r"$\chi^2$", lambda s: s.get('red_chi2')),
                            (r"$p$", lambda s: s.get('p_value')),
                            (r"$N_{\rm dof}$", lambda s: s.get('dof',0)),
                            (r"$N_{\rm cut}$", lambda s: s.get('dof', 0) - s.get('truncated_dof', 0)),
                            (r"$\chi^2_{\rm truncated}$", lambda s: s.get('truncated_red_chi2')),
                            (r"$p_{\rm truncated}$", lambda s: s.get('truncated_p_value')),
                        ]

                        # Populate Markdown Rows
                        for metric_label, metric_getter in metrics_map:
                            row = [metric_label]
                            for recon in recon_dicts:
                                score = recon.get('score', {})
                                val = metric_getter(score)
                                row.append(format_val(val))
                            md_lines.append("| " + " | ".join(row) + " |")

                        md_table = "\n".join(md_lines)
                        st.markdown(md_table)

            st.success("Reconstruction complete!")
