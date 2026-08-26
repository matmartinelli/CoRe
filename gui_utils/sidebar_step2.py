import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

def render_sidebar_step2():
    st.sidebar.header("Step 2: Reconstruction Settings")
    
    if st.sidebar.button("⬅️ Back to Data Upload"):
        st.session_state.step = 1
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Add Method to Comparison**")
    config_label = st.sidebar.text_input("Config Label", value=f"Config {len(st.session_state.recon_configs)+1}")
    method = st.sidebar.selectbox('Reconstruction method', ['Gaussian Process', 'Binned'])

    recon_entry = {"label": config_label, "method": method}

    # Per-method Evaluation Grid Settings
    st.sidebar.markdown("**Evaluation Grid Settings**")
    recon_entry["N"] = st.sidebar.number_input("Points (N)", min_value=1, max_value=500, value=15, step=1, key=f"N_{config_label}")
    recon_entry["xmin"] = st.sidebar.number_input("Minimum x", value=0.1, step=0.01, format="%.2f", key=f"xmin_{config_label}")
    recon_entry["xmax"] = st.sidebar.number_input("Maximum x", value=2.0, step=0.01, format="%.2f", key=f"xmax_{config_label}")

    # Extract all unique Y-labels across currently loaded datasets
    all_y_labels = []
    for config in st.session_state.data_store.values():
        for y_lbl in config["y_labels"]:
            if y_lbl not in all_y_labels:
                all_y_labels.append(y_lbl)

    fiducial_specs = {}

    if method == 'Binned':
        binning_method = st.sidebar.selectbox('Binning method', ['FLAT', 'GLS'])
        recon_entry["binning_method"] = binning_method

        if binning_method == 'FLAT':
            st.sidebar.markdown("**Fiducial Functions (FLAT)**")
            
            for y_lbl in all_y_labels:
                st.sidebar.caption(f"Fiducial for `{y_lbl}`:")
                fid_type = st.sidebar.radio(
                    f"Input type for {y_lbl}",
                    options=["Expression", "Tabulated File"],
                    key=f"fid_type_{config_label}_{y_lbl}",
                    horizontal=True
                )
                
                if fid_type == "Expression":
                    expr_str = st.sidebar.text_input(
                        f"Expression ({y_lbl})",
                        value="3-1*x+5*x**2-0.1*x**3",
                        key=f"fid_expr_{config_label}_{y_lbl}",
                        help="Enter math function in terms of 'x'."
                    )
                    fiducial_specs[y_lbl] = {"type": "expression", "val": expr_str}
                else:
                    tab_file = st.sidebar.file_uploader(
                        f"Upload Tabulated ({y_lbl})",
                        type=["txt", "csv"],
                        key=f"fid_file_{config_label}_{y_lbl}",
                        help="Upload file with header x,f"
                    )
                    fiducial_specs[y_lbl] = {"type": "file", "val": tab_file}

    elif method == 'Gaussian Process':
        kernel = st.sidebar.selectbox('Kernel type', ['RBF', 'MATERN3/2', 'MATERN5/2'])
        hp_method = st.sidebar.selectbox("Hyper-parameters search", ["MAP", "BAYESIAN"])
        recon_entry["kernel"] = kernel
        recon_entry["hp_method"] = hp_method

        if hp_method == 'BAYESIAN':
            sampler = st.sidebar.selectbox("Sampler", ["emcee", "nautilus"])
            sampsets = st.sidebar.selectbox("Sampler settings", ["poor", "good"])
            recon_entry["sampler"] = sampler
            recon_entry["sampsets"] = sampsets

        n_samples = st.sidebar.slider("Samples", min_value=10, max_value=100, value=50)
        recon_entry["n_samples"] = n_samples

    if st.sidebar.button("➕ Add Method"):
        valid = True
        fiducial_funcs = {}

        if method == 'Binned' and recon_entry.get("binning_method") == 'FLAT':
            for y_lbl, spec in fiducial_specs.items():
                if spec["type"] == "expression":
                    try:
                        f_test = eval(f"lambda x: {spec['val'].strip()}", {"__builtins__": None, "np": np, "numpy": np})
                        _ = f_test(1.0)
                        fiducial_funcs[y_lbl] = f_test
                    except Exception as e:
                        st.sidebar.error(f"Invalid expression for '{y_lbl}': {e}")
                        valid = False
                elif spec["type"] == "file":
                    file_obj = spec["val"]
                    if file_obj is None:
                        st.sidebar.error(f"Please upload a tabulated file for '{y_lbl}'.")
                        valid = False
                    else:
                        try:
                            tab_df = pd.read_csv(file_obj, sep=r'\s+', header=0)
                            x_tab = tab_df['x'].values
                            y_tab = tab_df['f'].values
                            
                            interp_func = interp1d(x_tab, y_tab, kind='linear', bounds_error=False, fill_value="extrapolate")
                            fiducial_funcs[y_lbl] = interp_func
                        except Exception as e:
                            st.sidebar.error(f"Error reading file for '{y_lbl}': {e}")
                            valid = False

            recon_entry["fiducial_funcs"] = fiducial_funcs

        if valid:
            st.session_state.recon_configs.append(recon_entry)
            st.sidebar.success(f"Added configuration: '{config_label}'")
            st.rerun()

    if st.session_state.recon_configs:
        st.sidebar.markdown("**Configured Methods to Compare:**")
        for idx, cfg in enumerate(st.session_state.recon_configs):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.caption(f"• {cfg['label']} ({cfg['method']})")
            if col2.button("❌", key=f"del_cfg_{idx}"):
                st.session_state.recon_configs.pop(idx)
                st.rerun()
