import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, block_diag

class BinnedCalculator:
    def __init__(self, df_data, df_cov, method='FLAT', chatty=True):
        """
        CoRe Engine for Joint Binned Non-Parametric Reconstruction of multiple functions
        with full cross-variable error propagation.
        
        df_data: pd.DataFrame containing 'x' and columns ['f','f_err'] or ['f1', 'f2', ..., 'f1_err', 'f2_err', ...]
        df_cov:  pd.DataFrame containing the full joint data covariance matrix (N_funcs*N_data x N_funcs*N_data)
        """
        self.chatty  = chatty
        self.method  = method
        self.df_data = df_data
        self.df_cov  = df_cov
        self.x_old   = df_data['x'].values

        self.func_cols = [c for c in self.df_data.columns if c != 'x' and not c.endswith('_err')]

    def _get_valid_mask(self, x_new):
        """Identifies indices of bins containing at least one data point."""
        midpoints = (x_new[:-1] + x_new[1:]) / 2.0
        bin_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
        mask = []
        for i in range(len(x_new)):
            in_bin = (self.x_old >= bin_edges[i]) & (self.x_old < bin_edges[i+1])
            mask.append(np.any(in_bin))
        return np.array(mask)

    def get_mask(self, x, i, lo_a, hi_a):
        n = len(x)
        return ((x >= lo_a[i]) & (x <  hi_a[i])) if i < n - 1 \
            else ((x >= lo_a[i]) & (x <= hi_a[i]))

    def gls_w(self, idx):
        cov = self.df_cov.values
        cf    = cho_factor(cov[np.ix_(idx, idx)])
        ones  = np.ones(idx.size)
        Cinv1 = cho_solve(cf, ones)
        return Cinv1 / (ones @ Cinv1)

    def centered_bao_windows(self, x_new):
        n = len(x_new)
        hw = np.zeros(n)
        for i in range(n):
            if   i == 0:   hw[i] = (x_new[1]  - x_new[0])  / 2
            elif i == n-1: hw[i] = (x_new[-1] - x_new[-2]) / 2
            else:          hw[i] = min(x_new[i]-x_new[i-1], x_new[i+1]-x_new[i]) / 2
        return x_new, x_new-hw, x_new+hw, hw

    def GLS_binning(self, x_new, min_per_bin=1, edge_optimize=True,
                    max_iter=50, tol=1e-5, label="value"):
        """
        GLS binning of a generic dataset with automatic filtering of faulty bins.
        Bins containing fewer than min_per_bin points are automatically discarded.
        """
        # We look at the first function column for finding data boundaries
        target_col = self.func_cols[0]
        val = self.df_data[target_col].values
        n   = len(x_new)
        _, lo, hi, _ = self.centered_bao_windows(x_new)
        lo, hi = lo.copy(), hi.copy()

        # --- Well-covered vs sparse -------------------------------------------
        N_init     = np.array([self.get_mask(self.x_old, i, lo, hi).sum() for i in range(n)])
        well       = N_init >= min_per_bin
        
        # Discard entirely empty bins right away to prevent edge optimization crashes
        valid_mask = N_init > 0
        well = well & valid_mask
        
        sparse_idx = np.where(~well & valid_mask)[0]
        empty_idx = np.where(~valid_mask)[0]
        
        if self.chatty:
            print(f"  [{label}] well-covered bins: {np.where(well)[0].tolist()}")
            if sparse_idx.size:
                print(f"  [{label}] sparse bins (kept but flagged): {sparse_idx.tolist()}")
            if empty_idx.size:
                print(f"  [{label}] empty bins (DISCARDED from calculation): {empty_idx.tolist()}")

        # --- Iterative edge optimization → z_eff = z_bao ----------------------
        converged = False
        ms        = np.inf
        if edge_optimize and well.any():
            for it in range(max_iter):
                x_eff_iter     = np.full(n, np.nan)
                lo_new, hi_new = lo.copy(), hi.copy()
                for i in range(n):
                    if not well[i]: continue
                    idx = np.where(self.get_mask(self.x_old, i, lo, hi))[0]
                    if idx.size == 0: continue
                    w             = self.gls_w(idx)
                    x_eff_iter[i] = w @ self.x_old[idx]
                    shift         = x_new[i] - x_eff_iter[i]
                    lo_new[i]    += shift
                    hi_new[i]    += shift
                    if i > 0:        lo_new[i] = max(lo_new[i], hi_new[i-1])
                    if i < n - 1:    hi_new[i] = min(hi_new[i], lo_new[i+1])
                ms = np.nanmax(np.abs(x_new[well] - x_eff_iter[well]))
                lo, hi = lo_new, hi_new
                if ms < tol:
                    converged = True
                    if self.chatty:
                        print(f"  [{label}] edge optimization converged at iter {it+1}  "
                              f"(max |x_eff - x_new| = {ms:.2e})")
                    break
            if not converged and self.chatty:
                print(f"  [{label}] WARNING: edge optimization did NOT converge "
                      f"in {max_iter} iters (max |x_eff - x_new| = {ms:.2e})")

        # --- Final binning + full covariance propagation on valid spaces ------
        # Identify how many valid bins survived
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        N_funcs = len(self.func_cols)
        
        # Build stacked outputs only for the valid bins
        f_new_stacked = np.zeros(N_funcs * n_valid)
        C_new_joint = np.zeros((N_funcs * n_valid, N_funcs * n_valid))

        # We need to pull from the full multi-variable structure
        N_old = len(self.x_old)
        labels_old = [f'{col}_{i}' for col in self.func_cols for i in range(N_old)]
        cov_full = self.df_cov.loc[labels_old, labels_old].values

        # Store weights and old mapping assignments for valid bins
        weights_per_func_bin = {}
        idx_per_func_bin = {}

        for f_idx, col in enumerate(self.func_cols):
            val_func = self.df_data[col].values
            
            for v_idx, i in enumerate(valid_indices):
                idx = np.where(self.get_mask(self.x_old, i, lo, hi))[0]
                idx_per_func_bin[(f_idx, v_idx)] = idx
                
                # Compute weights based on the original full covariance function block
                old_func_slice = slice(f_idx * N_old, (f_idx + 1) * N_old)
                cov_block = cov_full[old_func_slice, old_func_slice]
                
                cf = cho_factor(cov_block[np.ix_(idx, idx)])
                ones = np.ones(idx.size)
                Cinv1 = cho_solve(cf, ones)
                w = Cinv1 / (ones @ Cinv1)
                weights_per_func_bin[(f_idx, v_idx)] = w
                
                # Set stacked value
                f_new_stacked[f_idx * n_valid + v_idx] = w @ val_func[idx]

        # Populate cross-covariance matrix elements across valid variables and bins
        for f1_idx in range(N_funcs):
            for v1_idx in range(n_valid):
                idx1 = idx_per_func_bin[(f1_idx, v1_idx)]
                w1 = weights_per_func_bin[(f1_idx, v1_idx)]
                row_global_slice = f1_idx * N_old + idx1
                
                for f2_idx in range(N_funcs):
                    for v2_idx in range(n_valid):
                        idx2 = idx_per_func_bin[(f2_idx, v2_idx)]
                        w2 = weights_per_func_bin[(f2_idx, v2_idx)]
                        col_global_slice = f2_idx * N_old + idx2
                        
                        sub_cov = cov_full[np.ix_(row_global_slice, col_global_slice)]
                        val_cov = w1 @ sub_cov @ w2
                        
                        C_new_joint[f1_idx * n_valid + v1_idx, f2_idx * n_valid + v2_idx] = val_cov

        return f_new_stacked, C_new_joint, valid_mask

    def flat_residual_binning(self, x_new, fiducial):
        N_old  = len(self.x_old)
        N_bins_total = len(x_new)
        N_funcs = len(self.func_cols)

        # 1. Identify valid masks before setting up matrices
        valid_mask = self._get_valid_mask(x_new)
        self.x_valid = x_new[valid_mask]
        N_bins_valid = len(self.x_valid)

        if self.chatty and (N_bins_total - N_bins_valid > 0):
            empty_indices = np.where(~valid_mask)[0]
            print(f"BinnedCalculator: Discarding {N_bins_total - N_bins_valid} faulty bin(s) "
                  f"at indices {empty_indices} (centers: {x_new[empty_indices]}) because they contain no data points.")

        # Align old covariance matrix
        labels_old = [f'{col}_{i}' for col in self.func_cols for i in range(N_old)]
        cov_old_values = self.df_cov.loc[labels_old, labels_old].values

        # Build concatenated data and fiducial vectors
        f_old_list = []
        fid_f_old_list = []
        for idx, col in enumerate(self.func_cols):
            f_old_list.append(self.df_data[col].values)
            if isinstance(fiducial, (list, tuple)):
                fid_val = fiducial[idx](self.x_old)
            else:
                fid_val = fiducial(self.x_old)
            fid_f_old_list.append(fid_val)

        f_old_stacked = np.concatenate(f_old_list)
        fid_f_old_stacked = np.concatenate(fid_f_old_list)
        
        # Compute joint residuals
        self.residual_old = f_old_stacked / fid_f_old_stacked

        # Scale joint covariance matrix for flat residuals
        G = np.diag(1.0 / fid_f_old_stacked)
        self.C_residual_old = G @ cov_old_values @ G.T

        # Define bin boundaries using midpoint limits from the original full grid
        midpoints = (x_new[:-1] + x_new[1:]) / 2.0
        self.bin_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])

        # Build the single-variable mapping matrix A_single only for the VALID bins (N_old x N_bins_valid)
        A_single = np.zeros((N_old, N_bins_valid))
        for i, idx in enumerate(np.where(valid_mask)[0]):
            in_bin = (self.x_old >= self.bin_edges[idx]) & (self.x_old < self.bin_edges[idx+1])
            A_single[in_bin, i] = 1.0

        # Construct the joint multi-variable assignment matrix using block diagonal expansion
        A_joint = block_diag(*[A_single for _ in range(N_funcs)])

        # Joint Matrix compression via Cholesky decomposition solvers
        c, low = cho_factor(self.C_residual_old)
        invC_A = cho_solve((c, low), A_joint)

        # Compute compressed residual covariance matrix (Fisher Information inversion)
        Fisher_matrix = A_joint.T @ invC_A
        c_new, low_new = cho_factor(Fisher_matrix)
        self.C_residual_new = cho_solve((c_new, low_new), np.eye(N_funcs * N_bins_valid))

        # Direct solver for the compressed residuals
        rhs = invC_A.T @ self.residual_old
        self.residual_new = self.C_residual_new @ rhs

        # Project parameters back out to joint physical space for valid points
        fid_f_new_list = []
        for idx, col in enumerate(self.func_cols):
            if isinstance(fiducial, (list, tuple)):
                fid_val = fiducial[idx](self.x_valid)
            else:
                fid_val = fiducial(self.x_valid)
            fid_f_new_list.append(fid_val)
            
        fid_f_new_stacked = np.concatenate(fid_f_new_list)
        f_new_stacked = self.residual_new * fid_f_new_stacked

        # Scale binned joint matrix back to original physical units
        G_new = np.diag(fid_f_new_stacked)
        C_new = G_new @ self.C_residual_new @ G_new.T

        return f_new_stacked, C_new, valid_mask

    def get_derivatives(self, x_new, valid_mask, f_new_stacked, C_new):
        # Dynamically build operators based ONLY on the valid coordinates
        x_valid = x_new[valid_mask]
        N_bins = len(x_valid)
        N_funcs = len(self.func_cols)

        D1 = np.zeros((N_bins, N_bins))
        D2 = np.zeros((N_bins, N_bins))
        J  = np.zeros((N_bins, N_bins))
        I  = np.eye(N_bins)

        # Build standard Finite Difference Derivative Stencils on the valid coordinate mesh
        for i in range(N_bins):
            if i == 0:
                h = x_valid[1] - x_valid[0]
                D1[0, 0], D1[0, 1] = -1/h, 1/h
            elif i == N_bins - 1:
                h = x_valid[-1] - x_valid[-2]
                D1[-1, -2], D1[-1, -1] = -1/h, 1/h
            else:
                h_minus = x_valid[i] - x_valid[i-1]
                h_plus = x_valid[i+1] - x_valid[i]

                D1[i, i-1] = -h_plus / (h_minus * (h_minus + h_plus))
                D1[i, i]   = (h_plus - h_minus) / (h_minus * h_plus)
                D1[i, i+1] = h_minus / (h_plus * (h_minus + h_plus))

                D2[i, i-1] = 2 / (h_minus * (h_minus + h_plus))
                D2[i, i]   = -2 / (h_minus * h_plus)
                D2[i, i+1] = 2 / (h_plus * (h_minus + h_plus))

        # Explicit 3-point edge boundary corrections for second derivative
        if N_bins >= 3:
            h0, h1 = x_valid[1] - x_valid[0], x_valid[2] - x_valid[1]
            D2[0, 0], D2[0, 1], D2[0, 2] = 2/(h0*(h0+h1)), -2/(h0*h1), 2/(h1*(h0+h1))
            hn2, hn1 = x_valid[-2] - x_valid[-3], x_valid[-1] - x_valid[-2]
            D2[-1, -3], D2[-1, -2], D2[-1, -1] = 2/(hn2*(hn2+hn1)), -2/(hn2*hn1), 2/(hn1*(hn2+hn1))

        # Trapezoidal rule engine for the running integration matrix over valid points
        for i in range(1, N_bins):
            h = x_valid[i] - x_valid[i-1]
            J[i, :i] = J[i-1, :i]
            J[i, i-1] += 0.5 * h
            J[i, i]   += 0.5 * h

        # Boundary edge extrapolation correction if the grid does not originate at 0
        if x_valid[0] > 0:
            denom = x_valid[1] - x_valid[0]
            c0 = 0.5 * x_valid[0] * (2.0 * x_valid[1] - x_valid[0]) / denom
            c1 = -0.5 * x_valid[0] * x_valid[0] / denom
            J[:, 0] += c0
            J[:, 1] += c1

        # Evaluate physical observables across all functions simultaneously
        T_f  = np.eye(N_funcs * N_bins)
        T_d1 = block_diag(*[D1 for _ in range(N_funcs)])
        T_d2 = block_diag(*[D2 for _ in range(N_funcs)])
        T_J  = block_diag(*[J for _ in range(N_funcs)])

        f_new_vals   = T_f @ f_new_stacked
        d1_new_vals  = T_d1 @ f_new_stacked
        d2_new_vals  = T_d2 @ f_new_stacked
        int_new_vals = T_J @ f_new_stacked

        # =========================================================================
        # 3. JOINT COVARIANCE BLOCK MATRIX PROPAGATION (N_bins is now N_bins_valid)
        # =========================================================================
        T_joint = np.vstack([T_f, T_d1, T_d2, T_J])
        C_joint = T_joint @ C_new @ T_joint.T

        # Generate unique structured block label coordinates dynamically using only valid indices
        labels = []
        valid_idx_list = np.where(valid_mask)[0].tolist()
        
        for col in self.func_cols:
            labels.extend([f'{col}_{i}' for i in valid_idx_list])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'd1{suffix}_{i}' for i in valid_idx_list])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'd2{suffix}_{i}' for i in valid_idx_list])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'int{suffix}_{i}' for i in valid_idx_list])

        covmat_df = pd.DataFrame(C_joint, index=labels, columns=labels)

        # Isolate diagonal errors cleanly using labeled lookup mapping
        diag = np.sqrt(np.diag(C_joint))
        error_series = pd.Series(diag, index=labels)

        data_df = pd.DataFrame({'x': x_valid})

        # Unpack the values and map their corresponding standard errors
        for idx, col in enumerate(self.func_cols):
            suffix = col[1:] if col.startswith('f') else col
            slice_idx = slice(idx * N_bins, (idx + 1) * N_bins)
            
            # Populate reconstructed parameters
            data_df[col] = f_new_vals[slice_idx]
            data_df[f'd1{suffix}'] = d1_new_vals[slice_idx]
            data_df[f'd2{suffix}'] = d2_new_vals[slice_idx]
            data_df[f'int{suffix}'] = int_new_vals[slice_idx]
            
            # Map standard errors explicitly using true global bin labels to keep indexes correct
            data_df[f'{col}_err'] = [error_series[f'{col}_{i}'] for i in valid_idx_list]
            data_df[f'd1{suffix}_err'] = [error_series[f'd1{suffix}_{i}'] for i in valid_idx_list]
            data_df[f'd2{suffix}_err'] = [error_series[f'd2{suffix}_{i}'] for i in valid_idx_list]
            data_df[f'int{suffix}_err'] = [error_series[f'int{suffix}_{i}'] for i in valid_idx_list]

        # Explicitly enforce requested column sequence ordering
        col_order = ['x']
        for prefix in ['', 'd1', 'd2', 'int']:
            for col in self.func_cols:
                p = prefix + (col[1:] if col.startswith('f') else col) if prefix else col
                col_order.append(p)
        for prefix in ['', 'd1', 'd2', 'int']:
            for col in self.func_cols:
                p = prefix + (col[1:] if col.startswith('f') else col) if prefix else col
                col_order.append(f'{p}_err')

        data_df = data_df[col_order]

        return data_df, covmat_df

    def reconstruct(self,x_new,fiducial=None,tol=1.e-5,max_iter=50):
        """
        Main runner pipeline for the multi-variable Binned Non-Parametric reconstruction.
        """
        if fiducial is None:
            fiducial = lambda x: np.ones_like(x)
        
        if self.method == 'FLAT':
            f_new, C_new, valid_mask = self.flat_residual_binning(x_new, fiducial)
        elif self.method == 'GLS':
            f_new, C_new, valid_mask = self.GLS_binning(x_new,tol=tol,max_iter=max_iter)
        else:
            sys.exit('UNKNOWN BINNING METHOD: {}'.format(self.method))

        # Run derivative and error calculation using only valid indices
        data_df, covmat_df = self.get_derivatives(x_new, valid_mask, f_new, C_new)

        return data_df, covmat_df, valid_mask
