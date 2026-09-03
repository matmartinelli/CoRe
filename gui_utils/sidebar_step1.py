import pandas as pd
import streamlit as st

def render_sidebar_step1():
    st.sidebar.header("Step 1: Data Upload")

    with st.sidebar.form("dataset_upload_form", clear_on_submit=True):
        dataset_name  = st.text_input("Dataset Name", placeholder="e.g., CC_2026",help='This name will be used to refer to this dataset throughout the pipeline')
        dataset_label = st.text_input("Dataset Label", placeholder="e.g., My favourite dataset",help='This label will be used for plots')
        data_file     = st.file_uploader("Upload Dataset", type=["txt", "csv"])
        cov_file      = st.file_uploader("Upload Covariance Matrix", type=["txt", "csv"])

        st.markdown("**Axis Configuration**")
        x_label = st.text_input("X-axis Label", value="x")
        y_labels_input = st.text_input(
            "Y-axis Labels (comma-separated)",
            value="f(x)",
            help="Comma-separated column labels if file contains multiple functions."
        )

        add_dataset_btn = st.form_submit_button("Add Dataset Pair")

        if add_dataset_btn:
            if not dataset_name.strip():
                st.sidebar.error("Please provide a name for the dataset.")
            elif not data_file or not cov_file:
                st.sidebar.error("Both dataset and covariance files are required.")
            else:
                try:
                    data_df = pd.read_csv(data_file, sep=r'\s+', header=0)
                    cov_df  = pd.read_csv(cov_file, sep=r'\s+', index_col=0, header=0)
                    cov_df.index = cov_df.columns
                    parsed_y_labels = [y.strip() for y in y_labels_input.split(",") if y.strip()]

                    st.session_state.data_store[dataset_name.strip()] = {
                        "data_df": data_df,
                        "cov_df": cov_df,
                        "x_label": x_label.strip(),
                        "y_labels": parsed_y_labels,
                        "label": dataset_label
                    }
                    st.sidebar.success(f"Added: '{dataset_name}'")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error loading files: {e}")

    if st.session_state.data_store:
        st.sidebar.markdown("**Loaded Datasets:**")
        for name in list(st.session_state.data_store.keys()):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.caption(f"• {name}")
            if col2.button("❌", key=f"del_ds_{name}"):
                del st.session_state.data_store[name]
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Output Settings (Optional)**")
        st.session_state.outroot = st.sidebar.text_input(
            "Output Root Path (`outroot`)",
            value=st.session_state.get("outroot", ""),
            placeholder="e.g., output/run1",
            help="Path and root name for export files. Leave empty to skip saving."
        )

        st.sidebar.markdown("---")
        if st.sidebar.button("Proceed to Reconstruction Settings ➔", type="primary"):
            st.session_state.step = 2
            st.rerun()
