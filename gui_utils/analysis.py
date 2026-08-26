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
    # TODO: this probably messes covariance!!
    fcols = [col for col in data_df if col != 'x' and '_err' not in col]
    datasets_input = [{'type': func, 'df': data_df, 'cov': cov_df} for func in fcols]
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
