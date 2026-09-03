import os
import re
import sys
import time
import contextlib
import numpy as np
import pandas as pd
import streamlit as st

from CoRe.reconstruction_tools.Binning import BinnedCalculator
from CoRe.reconstruction_tools.GaussianProcess import GPCalculator
from CoRe.derived_function.reconstructor import DerivedFunction
from CoRe.utils.diagnostics import Scorer

EPSILON = 1.e-12


def clean_terminal_output(text: str) -> str:
    """Strips ANSI escape codes and processes carriage returns for clean log rendering."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)

    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if '\r' in line:
            segments = line.split('\r')
            processed_lines.append(segments[-1])
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)


@contextlib.contextmanager
def st_capture_output(code_placeholder, refresh_rate_sec=0.1):
    """Context manager to capture standard stdout/stderr into a Streamlit code element with UI throttling."""
    class OutputBuffer:
        def __init__(self, placeholder, refresh_interval):
            self.placeholder = placeholder
            self.refresh_interval = refresh_interval
            self.raw_buffer = ""
            self.last_update_time = 0.0

        def write(self, text):
            self.raw_buffer += text
            current_time = time.time()
            if current_time - self.last_update_time >= self.refresh_interval:
                self.flush_ui()
                self.last_update_time = current_time

        def flush_ui(self):
            cleaned_text = clean_terminal_output(self.raw_buffer)
            self.placeholder.code(cleaned_text, language="text")

        def flush(self):
            pass

    buffer = OutputBuffer(code_placeholder, refresh_rate_sec)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buffer, buffer
    try:
        yield
    finally:
        buffer.flush_ui()
        sys.stdout, sys.stderr = old_stdout, old_stderr


def run_single_reconstruction(
    cfg: dict, 
    data_df: pd.DataFrame, 
    cov_df: pd.DataFrame, 
    y_labels: list, 
    dataset_name: str,
    outroot: str = "",
    num_datasets: int = 1,
    eigen_trunc_factor: float = 1e-12
) -> dict:
    """Executes a single reconstruction method, calculates diagnostic scores, and exports results to disk."""
    m_type = cfg["method"]
    x_recon = np.linspace(cfg["xmin"], cfg["xmax"], cfg["N"])

    print(f"\n--- Running [{cfg['label']}] ({m_type}) on '{dataset_name}' ---")

    if m_type == 'Binned':
        bin_engine = BinnedCalculator(data_df, cov_df, method=cfg["binning_method"])
        fid_funcs_dict = cfg.get("fiducial_funcs", {})
        fid_func_input = fid_funcs_dict.get(dataset_name, None)

        means, cov, mask = bin_engine.reconstruct(x_recon, fiducial=fid_func_input)
        joint_cov = pd.DataFrame(
            cov.values + np.eye(len(cov)) * EPSILON, 
            columns=cov.columns, 
            index=cov.index
        )

    elif m_type == 'Gaussian Process':
        kwargs = {'n_samples': cfg["n_samples"]}
        if cfg["hp_method"] == 'BAYESIAN':
            kwargs.update({
                'sampler_name': cfg["sampler"], 
                'sampler_options': cfg["sampsets"]
            })

        gp = GPCalculator(data_df, cov_df, kernel_type=cfg["kernel"], chatty=True)
        means, joint_cov, lml, info = gp.reconstruct(x_recon, method=cfg["hp_method"], **kwargs)

    # Diagnostic scoring
    fcols = [col for col in data_df if col != 'x' and '_err' not in col]
    datasets_input = [{'type': fcols, 'df': data_df, 'cov': cov_df}]
    scorer = Scorer(means, joint_cov, chatty=False, eigen_trunc_factor=eigen_trunc_factor)
    score = scorer.score_against_data(datasets_input)

    # Export output files if outroot is non-empty
    if outroot and outroot.strip():
        clean_outroot = outroot.strip()
        out_dir = os.path.dirname(clean_outroot)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        safe_cfg_label = cfg['label'].replace(" ", "_")
        safe_ds_name = dataset_name.replace(" ", "_")

        if num_datasets > 1:
            base_filename = f"{clean_outroot}_{safe_ds_name}_{safe_cfg_label}"
        else:
            base_filename = f"{clean_outroot}_{safe_cfg_label}"

        means_file = f"{base_filename}_means.txt"
        covmat_file = f"{base_filename}_covmat.txt"

        means.to_csv(means_file, sep=' ', index=False)
        joint_cov.to_csv(covmat_file, sep=' ', index=True)

        print(f"Saved means to: {means_file}")
        print(f"Saved covmat to: {covmat_file}")

    return {
        'means': means,
        'covmat': joint_cov,
        'method': m_type,
        'label': cfg['label'],
        'score': score
    }


def run_derived_reconstruction(
    recon_dict: dict,
    cov_dict: dict,
    method_dict: dict,
    derived_logics: list,
    derived_names: list,
    outroot: str = "",
    cfg_label: str = "",
    x_recon: np.ndarray = None
):
    """Executes DerivedFunction calculation to combine reconstructions into derived quantities and saves results to disk."""
    print(f"\n--- Running Derived Function Analysis for '{derived_names}' ---")
    reconstructor = DerivedFunction(recon_dict, cov_dict, method_dict, chatty=True)
    derived_sample = reconstructor.run(derived_logics, derived_names)

    # Save outputs if outroot is provided
    if outroot and outroot.strip() and derived_sample is not None and x_recon is not None:
        clean_outroot = outroot.strip()
        out_dir = os.path.dirname(clean_outroot)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        safe_cfg_label = cfg_label.replace(" ", "_")

        all_pars = derived_sample.getParamNames().list()
        means = derived_sample.getMeans()
        vars_arr = derived_sample.getVars()
        full_cov = derived_sample.getCovMatrix()

        for d_name in derived_names:
            safe_d_name = d_name.replace(" ", "_")
            if safe_cfg_label:
                base_filename = f"{clean_outroot}_{safe_cfg_label}_{safe_d_name}"
            else:
                base_filename = f"{clean_outroot}_{safe_d_name}"

            indices = [i for i, par in enumerate(all_pars) if par.startswith(f"{d_name}_")]
            if indices:
                d_means = means[indices]
                d_errors = np.sqrt(vars_arr[indices])
                
                df_out = pd.DataFrame({
                    'x': x_recon[:len(d_means)],
                    'value': d_means,
                    'error': d_errors
                })

                means_file = f"{base_filename}_derived_means.txt"
                covmat_file = f"{base_filename}_derived_covmat.txt"

                df_out.to_csv(means_file, sep=' ', index=False)
                print(f"Saved derived means to: {means_file}")

                if full_cov is not None:
                    d_cov = full_cov[np.ix_(indices, indices)]
                    cov_cols = [f"{d_name}_{i}" for i in range(len(indices))]
                    df_cov = pd.DataFrame(d_cov, columns=cov_cols, index=cov_cols)
                    df_cov.to_csv(covmat_file, sep=' ', index=True)
                    print(f"Saved derived covmat to: {covmat_file}")

    return derived_sample
