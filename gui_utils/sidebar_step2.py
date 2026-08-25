import numpy as np
import streamlit as st

def render_sidebar_step2():
    st.sidebar.header("Step 2: Reconstruction Settings")
    
    if st.sidebar.button("⬅️ Back to Data Upload"):
        st.session_state.step = 1
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Evaluation Grid Settings**")
    N = st.sidebar.number_input("Points (N)", min_value=1, max_value=200, value=15, step=1)
    xmin = st.sidebar.number_input("Minimum x", value=0.1, step=0.01, format="%.2f")
    xmax = st.sidebar.number_input("Maximum x", value=2.0, step=0.01, format="%.2f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Add Method to Comparison**")
    config_label = st.sidebar.text_input("Config Label", value=f"Config {len(st.session_state.recon_configs)+1}")
    method = st.sidebar.selectbox('Reconstruction method', ['Gaussian Process', 'Binned'])

    recon_entry = {"label": config_label, "method": method}

    if method == 'Binned':
        binning_method = st.sidebar.selectbox('Binning method', ['FLAT', 'GLS'])
        recon_entry["binning_method"] = binning_method

        fiducial_str = ""
        if binning_method == 'FLAT':
            fiducial_str = st.sidebar.text_input(
                "Fiducial function",
                value="3-1*x+5*x**2-0.1*x**3",
                help="Expression in terms of 'x' using numpy functions."
            )
        recon_entry["fiducial_str"] = fiducial_str

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

        n_samples = st.sidebar.slider("Samples", min_value=10, max_value=500, value=100)
        recon_entry["n_samples"] = n_samples

    if st.sidebar.button("➕ Add Method"):
        valid = True
        if method == 'Binned' and recon_entry.get("binning_method") == 'FLAT':
            try:
                f_test = eval(f"lambda x: {fiducial_str.strip()}", {"__builtins__": None, "np": np, "numpy": np})
                _ = f_test(1.0)
                recon_entry["fiducial_func"] = f_test
            except Exception as e:
                st.sidebar.error(f"Invalid fiducial function syntax: {e}")
                valid = False

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

    return N, xmin, xmax
