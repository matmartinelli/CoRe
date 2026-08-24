import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from CoRe.reconstruction_tools.Binning         import BinnedCalculator
from CoRe.reconstruction_tools.GaussianProcess import GPCalculator
from CoRe.utils.diagnostics                    import Scorer


#Plotting
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc


rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

red    = '#8e001c'
yellow = '#ffb302'

sidelegend = {'bbox_to_anchor': (1.04,0.5),
              'loc': "center left",
              'frameon': False}
bottomlegend = {'bbox_to_anchor': (0.35,-0.2),
                'loc': "center left",
                'frameon': False,
                'ncols': 3}

st.title("CoRe Reconstruction Pipeline")

if "data_store" not in st.session_state:
    st.session_state.data_store = {}

# -----------------------------------------------------------------------------
# SIDEBAR: DATA UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.header("1. Data Upload")

# Form to collect name + data + covariance
with st.sidebar.form("dataset_upload_form", clear_on_submit=True):
    dataset_name = st.text_input("Dataset Name/Label", placeholder="e.g., CC_2026")
    data_file = st.file_uploader("Upload Dataset", type=["txt", "csv"])
    cov_file = st.file_uploader("Upload Covariance Matrix", type=["txt", "csv"])

    add_button = st.form_submit_button("Add Pair")

    if add_button:
        if not dataset_name.strip():
            st.sidebar.error("Please provide a name for the dataset.")
        elif not data_file or not cov_file:
            st.sidebar.error("Both dataset and covariance matrix files are required.")
        else:
            st.session_state.data_store[dataset_name.strip()] = {
                "data": data_file,
                "cov": cov_file
            }
            st.sidebar.success(f"Added: '{dataset_name}'")

# Display loaded datasets with remove options
if st.session_state.data_store:
    st.sidebar.markdown("**Loaded Datasets:**")
    for name in list(st.session_state.data_store.keys()):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.caption(f"• {name}")
        if col2.button("❌", key=f"del_{name}"):
            del st.session_state.data_store[name]
            st.rerun()

# -----------------------------------------------------------------------------
# SIDEBAR: RECONSTRUCTION SETTINGS
# -----------------------------------------------------------------------------
st.sidebar.header("2. Reconstruction Settings")

N    = st.sidebar.number_input("Number of reconstruction points",min_value=1,max_value=50,value=5,step=1)
xmin = st.sidebar.number_input("Minimum x value",value=0.1,step=0.01,format="%.2f")
xmax = st.sidebar.number_input("Maximum x value",value=2.0,step=0.01,format="%.2f")

method = st.sidebar.selectbox('Reconstruction method',['Binned','Gaussian Process'])
if method == 'Binned':
    binning_method = st.sidebar.selectbox('Binning method',['FLAT','GLS'])
elif method == 'Gaussian Process':
    kernel    = st.sidebar.selectbox('Kernel type',['RBF','MATERN3/2','MATERN5/2'])
    hp_method = st.sidebar.selectbox("Hyper-parameters seatch", ["MAP", "BAYESIAN"])
    if hp_method == 'BAYESIAN':
        sampler  = st.sidebar.selectbox("Sampler",["emcee","nautilus"])
        sampsets = st.sidebar.selectbox("Sampler settings",["poor","good"])
    n_samples = st.sidebar.slider("Samples", min_value=10, max_value=500, value=100)


# -----------------------------------------------------------------------------
# MAIN APP BODY
# -----------------------------------------------------------------------------
if st.session_state.data_store:
    if st.button("Run Reconstruction"):
        st.write(f"### Running `{method}` pipeline...")

        for name, files in st.session_state.data_store.items():
            files["data"].seek(0)
            files["cov"].seek(0)

            data_df = pd.read_csv(files["data"], sep=r'\s+')
            cov_df  = pd.read_csv(files["cov"], sep=r'\s+', index_col=0)

            x_recon = np.linspace(xmin,xmax,N)

            if method == 'Binned':
                print('bleah')
            elif method == 'Gaussian Process':
                kwargs = {'n_samples': n_samples}
                if hp_method == 'BAYESIAN':
                    kwargs = kwargs | {'sampler_name': sampler, 'sampler_options': sampsets}
                gp = GPCalculator(data_df,cov_df,kernel_type=kernel,chatty=True)
                means, joint_cov, lml, info = gp.reconstruct(x_recon,method=hp_method,**kwargs)

            # Place calculator calls here using df and cov_df
            xplot = np.linspace(min(data_df['x']),max(data_df['x']),100)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(x_recon,means['f']+means['f_err'],means['f']-means['f_err'],alpha=0.2,color=red)
            ax.plot(x_recon,means['f'],color=red)
            ax.errorbar(data_df['x'], data_df['f'], yerr=data_df['f_err'], fmt='o', label='Data', color='black')

            # Plot reconstruction (example using dummy arrays or your GP outputs)
            # ax.plot(x_recon, means, label='GP Reconstruction', color='red')
            # ax.fill_between(x_recon, means - std, means + std, alpha=0.3, color='red')

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Reconstruction: {name}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)

        st.success("Reconstruction complete!")
else:
    st.info("Please upload at least one dataset and covariance matrix pair using the sidebar.")
