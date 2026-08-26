import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
                            num_datasets=num_datasets
                        )
                        recon_dicts.append(recon_res)

                fig_res = plot_observable_recon(data_df, name, recon_dicts, x_label, y_labels)

            st.success("Reconstruction complete!")
