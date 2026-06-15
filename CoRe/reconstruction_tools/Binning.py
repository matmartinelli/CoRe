import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, block_diag

class BinnedCalculator:
    def __init__(self, df_data, df_cov, method='FLAT', chatty=True):
        """
        CoRe Engine for Joint Binned Non-Parametric Reconstruction of multiple functions
        with full cross-variable error propagation.
        
        df_data: pd.DataFrame containing 'x' and columns ['f1', 'f2', ..., 'f1_err', 'f2_err', ...]
        df_cov:  pd.DataFrame containing the full joint data covariance matrix (N_funcs*N_data x N_funcs*N_data)
        """
        self.chatty  = chatty
        self.method  = method
        self.df_data = df_data
        self.df_cov  = df_cov
        self.x_old   = df_data['x'].values

        self.func_cols = [c for c in self.df_data.columns if c != 'x' and not c.endswith('_err')]

    def get_mask(self,x,i,lo_a,hi_a):
        n = len(x)
        return ((x >= lo_a[i]) & (x <  hi_a[i])) if i < n - 1 \
            else ((x >= lo_a[i]) & (x <= hi_a[i]))

    def gls_w(self,idx):
        cov = self.df_cov.values
        cf    = cho_factor(cov[np.ix_(idx, idx)])
        ones  = np.ones(idx.size)
        Cinv1 = cho_solve(cf, ones)
        return Cinv1 / (ones @ Cinv1)

    def centered_bao_windows(self,x_new):
        n = len(x_new)
        hw = np.zeros(n)
        for i in range(n):
            if   i == 0:   hw[i] = (x_new[1]  - x_new[0])  / 2
            elif i == n-1: hw[i] = (x_new[-1] - x_new[-2]) / 2
            else:          hw[i] = min(x_new[i]-x_new[i-1], x_new[i+1]-x_new[i]) / 2
        return x_new, x_new-hw, x_new+hw, hw

    def GLS_binning(self,x_new,min_per_bin=1, edge_optimize=True,
                    max_iter=50, tol=1e-5, label="value"):
        """
        GLS binning of a generic [z, value] dataset at the BAO redshift grid,
        with iterative edge optimization that drives z_eff → z_bao for
        well-covered bins. No fiducial correction is applied — the binning is
        a purely linear, model-independent operation.

        For sparse bins (fewer than min_per_bin points), edge optimization
        cannot physically shift the GLS-weighted redshift, so z_eff is
        LABELED z_bao by convention and the residual offset is printed as a
        diagnostic. Decide downstream whether to mask such bins.

        Use cases
        ---------
        SN  (d_L) : min_per_bin=9, edge_optimize=True   →  converges to z_bao
        CC  (1/H) : min_per_bin=1, edge_optimize=True   →  no-op for 1-pt bins;
                                                           sparse bins flagged

        Parameters
        ----------
        data         : (N, 2)   [z_k, value_k]
        cov          : (N, N)   covariance of the values
        z_bao        : (n,)     target BAO grid
        min_per_bin  : threshold for "well-covered" vs "sparse"
        edge_optimize: iterate bin edges so GLS-effective z equals z_bao
        max_iter,tol : convergence parameters
        label        : tag used in printouts

        Returns
        -------
        dict with keys:
          z_bao
          z_eff        : true GLS centroid (well-covered) / z_bao (sparse)
          N            : count per bin
          val_bin, sig_val_bin, cov_val_bin
          lo, hi       : final bin edges
          well         : bool array, well-covered bins
          converged    : bool, whether edge optimization converged
          max_offset   : max |z_eff - z_bao| over well-covered bins after binning
        """
        val = self.df_data['f']
        n   = len(x_new)
        _, lo, hi, _ = self.centered_bao_windows(x_new)
        lo, hi = lo.copy(), hi.copy()

        # --- Well-covered vs sparse -------------------------------------------
        N_init     = np.array([self.get_mask(self.x_old,i, lo, hi).sum() for i in range(n)])
        well       = N_init >= min_per_bin
        sparse_idx = np.where(~well)[0]
        print(f"  [{label}] well-covered: {np.where(well)[0].tolist()}"
              + (f"   sparse: {sparse_idx.tolist()}" if sparse_idx.size else ""))

        # --- Iterative edge optimization → z_eff = z_bao ----------------------
        converged = False
        ms        = np.inf
        if edge_optimize and well.any():
            for it in range(max_iter):
                x_eff_iter     = np.full(n, np.nan)
                lo_new, hi_new = lo.copy(), hi.copy()
                for i in range(n):
                    if not well[i]: continue
                    idx = np.where(self.get_mask(self.x_old,i, lo, hi))[0]
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
                    print(f"  [{label}] edge optimization converged at iter {it+1}  "
                          f"(max |x_eff - x_new| = {ms:.2e})")
                    break
            if not converged:
                print(f"  [{label}] WARNING: edge optimization did NOT converge "
                      f"in {max_iter} iters (max |x_eff - x_new| = {ms:.2e})")

        # --- Final binning + full covariance propagation ----------------------
        weights = [None] * n
        indices = [None] * n
        val_bin = np.full(n, np.nan)
        x_eff   = x_new.copy()
        N_arr   = np.zeros(n, int)

        for i in range(n):
            idx       = np.where(self.get_mask(self.x_old,i, lo, hi))[0]
            N_arr[i]  = idx.size
            if idx.size == 0: continue
            indices[i] = idx
            w          = self.gls_w(idx)
            weights[i] = w
            val_bin[i] = w @ val[idx]
            # z_eff: true centroid for well-covered, labeled z_bao for sparse
            x_eff[i]   = (w @ self.x_old[idx]) if well[i] else x_new[i]

        cov = self.df_cov.values
        cov_val_bin = np.zeros((n, n))
        for i in range(n):
            if indices[i] is None: continue
            for j in range(i, n):
                if indices[j] is None: continue
                v = weights[i] @ cov[np.ix_(indices[i], indices[j])] @ weights[j]
                cov_val_bin[i, j] = v
                cov_val_bin[j, i] = v
        sig_val_bin = np.sqrt(np.diag(cov_val_bin))

        # --- Diagnostics on residual z_eff offsets ----------------------------
        max_offset = np.max(np.abs(x_eff[well] - x_new[well])) if well.any() else 0.0
        print(f"  [{label}] max |x_eff - x_bao| in well-covered bins: {max_offset:.2e}")
        for i in sparse_idx:
            if indices[i] is None: continue
            true_cen = weights[i] @ self.x_old[indices[i]]
            print(f"  [{label}] sparse bin {i}: x_new={x_new[i]:.4f}, "
                  f"true centroid={true_cen:.4f}, offset={true_cen-x_new[i]:+.4f}")


        return val_bin,cov_val_bin

    def flat_residual_binning(self, x_new, fiducial):
        N_old  = len(self.x_old)
        N_bins = len(x_new)
        N_funcs = len(self.func_cols)
        self.x_new = x_new

        # =========================================================================
        # 1. MULTI-VARIABLE FLAT RESIDUAL BINNING
        # =========================================================================
        # Align the old covariance matrix rows/columns to match function order
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

        # Scale joint covariance matrix for flat residuals: C_y = G @ C_old @ G^T
        G = np.diag(1.0 / fid_f_old_stacked)
        self.C_residual_old = G @ cov_old_values @ G.T

        # Define bin boundaries using midpoint limits
        midpoints = (x_new[:-1] + x_new[1:]) / 2.0
        self.bin_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])

        # Build the single-variable mapping matrix A_single (N_old x N_bins)
        A_single = np.zeros((N_old, N_bins))
        for i in range(N_bins):
            in_bin = (self.x_old >= self.bin_edges[i]) & (self.x_old < self.bin_edges[i+1])
            A_single[in_bin, i] = 1.0

        # Safety Check for Empty Bins
        points_per_bin = np.sum(A_single, axis=0)
        if np.any(points_per_bin == 0):
            empty_indices = np.where(points_per_bin == 0)[0]
            raise ValueError(f"Binning failed: Bins at indices {empty_indices} "
                             f"(centers: {x_new[empty_indices]}) contain no data points!")

        # Construct the joint multi-variable assignment matrix using block diagonal expansion
        A_joint = block_diag(*[A_single for _ in range(N_funcs)])

        # Joint Matrix compression via Cholesky decomposition solvers
        c, low = cho_factor(self.C_residual_old)
        invC_A = cho_solve((c, low), A_joint)  # Equivalent to: C_inv @ A_joint

        # Compute compressed residual covariance matrix (Fisher Information inversion)
        Fisher_matrix = A_joint.T @ invC_A
        c_new, low_new = cho_factor(Fisher_matrix)
        self.C_residual_new = cho_solve((c_new, low_new), np.eye(N_funcs * N_bins))

        # Direct solver for the compressed residuals
        rhs = invC_A.T @ self.residual_old
        self.residual_new = self.C_residual_new @ rhs

        # Project parameters back out to joint physical space
        fid_f_new_list = []
        for idx, col in enumerate(self.func_cols):
            if isinstance(fiducial, (list, tuple)):
                fid_val = fiducial[idx](x_new)
            else:
                fid_val = fiducial(x_new)
            fid_f_new_list.append(fid_val)
            
        fid_f_new_stacked = np.concatenate(fid_f_new_list)
        f_new_stacked = self.residual_new * fid_f_new_stacked

        # Scale binned joint matrix back to original physical units
        G_new = np.diag(fid_f_new_stacked)
        C_new = G_new @ self.C_residual_new @ G_new.T

        return f_new_stacked, C_new

    def get_derivatives(self, x_new, f_new_stacked, C_new):
        N_bins = len(x_new)
        N_funcs = len(self.func_cols)

        D1 = np.zeros((N_bins, N_bins))
        D2 = np.zeros((N_bins, N_bins))
        J  = np.zeros((N_bins, N_bins))
        I = np.eye(N_bins)

        # Build standard Finite Difference Derivative Stencils
        for i in range(N_bins):
            if i == 0:
                h = x_new[1] - x_new[0]
                D1[0, 0], D1[0, 1] = -1/h, 1/h
            elif i == N_bins - 1:
                h = x_new[-1] - x_new[-2]
                D1[-1, -2], D1[-1, -1] = -1/h, 1/h
            else:
                h_minus = x_new[i] - x_new[i-1]
                h_plus = x_new[i+1] - x_new[i]

                D1[i, i-1] = -h_plus / (h_minus * (h_minus + h_plus))
                D1[i, i]   = (h_plus - h_minus) / (h_minus * h_plus)
                D1[i, i+1] = h_minus / (h_plus * (h_minus + h_plus))

                D2[i, i-1] = 2 / (h_minus * (h_minus + h_plus))
                D2[i, i]   = -2 / (h_minus * h_plus)
                D2[i, i+1] = 2 / (h_plus * (h_minus + h_plus))

        # Explicit 3-point edge boundary corrections for second derivative
        if N_bins >= 3:
            h0, h1 = x_new[1] - x_new[0], x_new[2] - x_new[1]
            D2[0, 0], D2[0, 1], D2[0, 2] = 2/(h0*(h0+h1)), -2/(h0*h1), 2/(h1*(h0+h1))
            hn2, hn1 = x_new[-2] - x_new[-3], x_new[-1] - x_new[-2]
            D2[-1, -3], D2[-1, -2], D2[-1, -1] = 2/(hn2*(hn2+hn1)), -2/(hn2*hn1), 2/(hn1*(hn2+hn1))

        # Trapezoidal rule engine for the running integration matrix
        for i in range(1, N_bins):
            h = x_new[i] - x_new[i-1]
            J[i, :i] = J[i-1, :i]
            J[i, i-1] += 0.5 * h
            J[i, i]   += 0.5 * h

        # Boundary edge extrapolation correction if the grid does not originate at 0
        if x_new[0] > 0:
            denom = x_new[1] - x_new[0]
            c0 = 0.5 * x_new[0] * (2.0 * x_new[1] - x_new[0]) / denom
            c1 = -0.5 * x_new[0] * x_new[0] / denom
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
        # 3. JOINT COVARIANCE BLOCK MATRIX PROPAGATION
        # =========================================================================
        T_joint = np.vstack([T_f, T_d1, T_d2, T_J])
        C_joint = T_joint @ C_new @ T_joint.T

        # Generate unique structured block label coordinates dynamically
        labels = []
        for col in self.func_cols:
            labels.extend([f'{col}_{i}' for i in range(N_bins)])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'd1{suffix}_{i}' for i in range(N_bins)])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'd2{suffix}_{i}' for i in range(N_bins)])
            
        for col in self.func_cols:
            suffix = col[1:] if col.startswith('f') else col
            labels.extend([f'int{suffix}_{i}' for i in range(N_bins)])

        covmat_df = pd.DataFrame(C_joint, index=labels, columns=labels)

        # Isolate diagonal errors cleanly using labeled lookup mapping
        diag = np.sqrt(np.diag(C_joint))
        error_series = pd.Series(diag, index=labels)

        data_df = pd.DataFrame({'x': x_new})

        # Unpack the values and map their corresponding standard errors
        for idx, col in enumerate(self.func_cols):
            suffix = col[1:] if col.startswith('f') else col
            slice_idx = slice(idx * N_bins, (idx + 1) * N_bins)
            
            # Populate reconstructed parameters
            data_df[col] = f_new_vals[slice_idx]
            data_df[f'd1{suffix}'] = d1_new_vals[slice_idx]
            data_df[f'd2{suffix}'] = d2_new_vals[slice_idx]
            data_df[f'int{suffix}'] = int_new_vals[slice_idx]
            
            # Map standard errors explicitly using labels to prevent index shifting bugs
            data_df[f'{col}_err'] = [error_series[f'{col}_{i}'] for i in range(N_bins)]
            data_df[f'd1{suffix}_err'] = [error_series[f'd1{suffix}_{i}'] for i in range(N_bins)]
            data_df[f'd2{suffix}_err'] = [error_series[f'd2{suffix}_{i}'] for i in range(N_bins)]
            data_df[f'int{suffix}_err'] = [error_series[f'int{suffix}_{i}'] for i in range(N_bins)]

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

    def reconstruct(self, x_new, fiducial=None):
        """
        Main runner pipeline for the multi-variable Binned Non-Parametric reconstruction.
        """
        # Automatically isolate target function columns from errors
        #func_cols = [c for c in self.df_data.columns if c != 'x' and not c.endswith('_err')]

        if fiducial is None:
            fiducial = lambda x: np.ones_like(x)
        
        if self.method == 'FLAT':
            f_new, C_new = self.flat_residual_binning(x_new, fiducial)
        elif self.method == 'GLS':
            f_new, C_new = self.GLS_binning(x_new)
        else:
            sys.exit('UNKNOWN BINNING METHOD: {}'.format(self.method))

        # Fixed instance invocation logic (removed explicit self pass)
        data_df, covmat_df = self.get_derivatives(x_new, f_new, C_new)

        return data_df, covmat_df
