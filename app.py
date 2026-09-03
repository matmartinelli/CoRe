import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

from gui_utils.plots import (
    plot_data, 
    plot_covmat, 
    plot_observable_recon,
    plot_derived_triangle,
    plot_derived_summary
)
from gui_utils.sidebar_step1 import render_sidebar_step1
from gui_utils.sidebar_step2 import render_sidebar_step2
from gui_utils.sidebar_step3 import render_sidebar_step3
from gui_utils.analysis import (
    st_capture_output, 
    run_single_reconstruction, 
    run_derived_reconstruction
)

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
if "last_recon_results" not in st.session_state:
    st.session_state.last_recon_results = {}


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
            data_df    = config["data_df"]
            cov_df     = config["cov_df"] 
            x_label    = config["x_label"]
            y_labels   = config["y_labels"]
            data_label = config["label"]

            with st.expander(f"📊 Dataset Preview: `{data_label}`", expanded=True):
                col_tbl, col_plt = st.columns([1, 1])
                with col_tbl:
                    st.markdown("### Raw Data Plot:")
                    plot_data(data_df, data_label, x_label, y_labels)

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

            st.session_state.last_recon_results = {}

            for name, config in st.session_state.data_store.items():
                data_df    = config["data_df"]
                cov_df     = config["cov_df"]
                x_label    = config["x_label"]
                y_labels   = config["y_labels"]
                data_label = config["label"]

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

                st.session_state.last_recon_results[name] = recon_dicts

                col_plot, col_diag = st.columns([1.1, 1])

                with col_plot:
                    st.markdown("### Reconstruction Plot")
                    fig_res = plot_observable_recon(data_df, data_label, recon_dicts, x_label, y_labels)

                with col_diag:
                    st.markdown("### Diagnostic Scores")

                    if recon_dicts:
                        headers = ["Metric"] + [
                            f"{get_status_badge(r.get('score', {}).get('truncated_p_value', 1.0))} **{r['label']}**"
                            for r in recon_dicts
                        ]
                        
                        md_lines = ["| " + " | ".join(headers) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                        metrics_map = [
                            (r"$\chi^2$", lambda s: s.get('red_chi2')),
                            (r"$p$", lambda s: s.get('p_value')),
                            (r"$N_{\rm dof}$", lambda s: s.get('dof', 0)),
                            (r"$N_{\rm cut}$", lambda s: s.get('dof', 0) - s.get('truncated_dof', 0)),
                            (r"$\chi^2_{\rm truncated}$", lambda s: s.get('truncated_red_chi2')),
                            (r"$p_{\rm truncated}$", lambda s: s.get('truncated_p_value')),
                        ]

                        for metric_label, metric_getter in metrics_map:
                            row = [metric_label]
                            for recon in recon_dicts:
                                score = recon.get('score', {})
                                val = metric_getter(score)
                                row.append(format_val(val))
                            md_lines.append("| " + " | ".join(row) + " |")

                        md_table = "\n".join(md_lines)
                        st.markdown(md_table)

        if st.session_state.last_recon_results:
            st.success("Reconstruction complete!")
            st.markdown("---")
            if st.button("➡️ Proceed to Derived Quantities (Step 3)", type="primary"):
                st.session_state.step = 3
                st.rerun()

elif st.session_state.step == 3:
    render_sidebar_step3()

    st.title("CoRe Reconstruction Pipeline")
    st.subheader("Step 3: Combine Reconstructions for Derived Quantities")

    if not st.session_state.last_recon_results or not st.session_state.recon_configs:
        st.warning("⚠️ No reconstruction results found. Please run Step 2 reconstructions first.")
    elif not st.session_state.get("derived_funcs", []):
        st.warning("⚠️ Please add at least one derived function in the sidebar.")
    else:
        num_configs = len(st.session_state.recon_configs)
        ds_names = list(st.session_state.data_store.keys())
        first_ds_cfg = st.session_state.recon_configs[0]
        x_recon = np.linspace(first_ds_cfg["xmin"], first_ds_cfg["xmax"], first_ds_cfg["N"])
        
        first_ds_key = list(st.session_state.data_store.keys())[0]
        x_label = st.session_state.data_store[first_ds_key].get("x_label", "x")
        outroot = st.session_state.get("outroot", "")

        st.info(
            f"Ready to compute **{len(st.session_state.derived_funcs)}** derived quantity/quantities "
            f"across **{num_configs}** reconstruction method(s) for dataset names: `{ds_names}`."
        )

        if st.button("🚀 Run Derived Function Analysis", type="primary"):
            method_dict = st.session_state.get("derived_method_dict", {})
            derived_funcs = st.session_state.get("derived_funcs", [])

            derived_logics = []
            derived_names = []
            derived_tex_map = {}

            valid_setup = True
            for df_item in derived_funcs:
                d_name = df_item["name"]
                d_logic_str = df_item["logic_str"]
                d_tex = df_item.get("tex_label", d_name)

                try:
                    f_logic = eval(d_logic_str, {"__builtins__": None, "np": np, "numpy": np})
                    derived_logics.append(f_logic)
                    derived_names.append(d_name)
                    derived_tex_map[d_name] = d_tex
                except Exception as e:
                    st.error(f"Invalid derived logic syntax for '{d_name}': {e}")
                    valid_setup = False

            if valid_setup and derived_logics:
                all_derived_results = {}

                for cfg_idx, cfg in enumerate(st.session_state.recon_configs):
                    cfg_label = cfg["label"]
                    st.markdown("---")
                    st.markdown(f"### Method: `{cfg_label}` ({cfg['method']})")

                    recon_dict = {}
                    cov_dict = {}

                    for ds_idx, ds_name in enumerate(ds_names, start=1):
                        alias = f"D{ds_idx}"
                        ds_recon = st.session_state.last_recon_results[ds_name][cfg_idx]

                        recon_dict[alias] = ds_recon['means']
                        cov_dict[alias] = ds_recon['covmat']
                        recon_dict[ds_name] = ds_recon['means']
                        cov_dict[ds_name] = ds_recon['covmat']

                    with st.expander(f"📺 Live Logs: `{cfg_label}`", expanded=False):
                        log_placeholder = st.empty()

                    try:
                        with st_capture_output(log_placeholder):
                            derived_sample = run_derived_reconstruction(
                                recon_dict=recon_dict,
                                cov_dict=cov_dict,
                                method_dict=method_dict,
                                derived_logics=derived_logics,
                                derived_names=derived_names,
                                outroot=outroot,
                                cfg_label=cfg_label,
                                x_recon=x_recon
                            )

                        grid_x = recon_dict[list(recon_dict.keys())[0]]['x'].values

                        derived_res = {
                            'sample': derived_sample,
                            'x_recon': grid_x
                        }

                        all_derived_results[cfg_label] = derived_res
                        st.success(f"Completed derived reconstructions for `{cfg_label}`!")

                        if derived_sample is not None:
                            st.markdown("#### GetDist Triangle Plot")
                            fig_tri = plot_derived_triangle(
                                derived_res=derived_sample,
                                derived_names=derived_names,
                                x_recon=grid_x,
                                color=cfg.get('color', 'blue'),
                                label=cfg_label
                            )
                            st.pyplot(fig_tri)

                    except Exception as e:
                        st.error(f"Error computing derived functions for configuration '{cfg_label}': {e}")

                st.session_state.derived_results = all_derived_results

                if all_derived_results:
                    st.markdown("---")
                    st.markdown("## Overall Derived Function Comparisons")
                    
                    for d_name in derived_names:
                        d_tex = derived_tex_map.get(d_name, d_name)
                        st.markdown(f"### Derived Function: `{d_name}`")
                        fig_summary = plot_derived_summary(
                            all_derived_results=all_derived_results,
                            recon_configs=st.session_state.recon_configs,
                            x_label=x_label,
                            derived_name=d_name,
                            tex_label=d_tex
                        )
                        st.pyplot(fig_summary)
