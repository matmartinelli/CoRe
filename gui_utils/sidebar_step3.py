import streamlit as st

def render_sidebar_step3():
    st.sidebar.header("Step 3: Derived Function Settings")

    # Method Selection
    method_type = st.sidebar.selectbox(
        "Derived Method Type",
        options=["realizations", "covariance"],
        index=0,
        help="Choose the algorithm type used to calculate standard deviations/covariances."
    )

    method_options = {}

    if method_type == "realizations":
        n_reals = st.sidebar.number_input(
            "Nreals (Number of realizations)",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            help="Number of realizations generated during derived sampling."
        )
        n_samples = st.sidebar.number_input(
            "Nsamples (Number of samples)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Number of samples per realization."
        )
        method_options = {
            "Nreals": int(n_reals),
            "Nsamples": int(n_samples)
        }

    st.session_state.derived_method_dict = {
        "type": method_type,
        "options": method_options
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Derived Functions Setup")

    # Initialize derived_funcs list if empty
    if "derived_funcs" not in st.session_state or not st.session_state.derived_funcs:
        st.session_state.derived_funcs = [
            {
                "name": "ratio_D1_D2",
                "logic_str": "lambda D1, D2: D1 / D2",
                "tex_label": r"D_1 / D_2"
            }
        ]

    # Add new derived function form
    with st.sidebar.form("add_derived_func_form", clear_on_submit=True):
        st.markdown("**Add Derived Function**")
        d_name = st.text_input("Quantity Name", placeholder="e.g. ratio_D1_D2")
        d_logic = st.text_input("Lambda Expression", placeholder="e.g. lambda D1, D2: D1 / D2")
        d_tex = st.text_input("TeX Label", placeholder=r"e.g. D_1 / D_2")
        
        add_btn = st.form_submit_button("➕ Add Derived Function")

        if add_btn:
            if not d_name.strip() or not d_logic.strip():
                st.sidebar.error("Please provide both a quantity name and a lambda expression.")
            else:
                st.session_state.derived_funcs.append({
                    "name": d_name.strip(),
                    "logic_str": d_logic.strip(),
                    "tex_label": d_tex.strip() if d_tex.strip() else d_name.strip()
                })
                st.rerun()

    # List configured functions
    if st.session_state.derived_funcs:
        st.sidebar.markdown("**Configured Derived Functions:**")
        for idx, df_item in enumerate(st.session_state.derived_funcs):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.caption(f"• **{df_item['name']}**: `{df_item['logic_str']}`")
            if col2.button("❌", key=f"del_dfunc_{idx}"):
                st.session_state.derived_funcs.pop(idx)
                st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Back to Step 2"):
        st.session_state.step = 2
        st.rerun()
