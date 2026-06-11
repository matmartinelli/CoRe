import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

class BinnedCalculator:
    def __init__(self,df_data,df_cov,method='FLAT',chatty=True):
        """
        CoRe Engine for Binned Non-Parametric Reconstruction with error propagation.
        
        df_data: pd.DataFrame containing columns ['x', 'f', 'f_err']
        df_cov:  pd.DataFrame containing the full data covariance matrix (N_data x N_data)
        """
        self.chatty  = chatty
        self.method  = method
        self.x_old   = df_data['x'].values
        self.f_old   = df_data['f'].values
        self.cov_old = df_cov.values if isinstance(df_cov, pd.DataFrame) else df_cov

    def flat_residual_binning(self,x_new,fiducial):

        N_old  = len(self.x_old)
        N_bins = len(x_new)
        self.x_new = x_new

        # =========================================================================
        # 1. FLAT RESIDUAL BINNING
        # =========================================================================
        fid_f_old = fiducial(self.x_old)
        self.residual_old = self.f_old / fid_f_old

        # Scale covariance matrix for flat residuals: C_y = G @ C_old @ G^T
        G = np.diag(1.0 / fid_f_old)
        self.C_residual_old = G @ self.cov_old @ G.T

        # Define bin boundaries using midpoint limits
        midpoints = (x_new[:-1] + x_new[1:]) / 2.0
        self.bin_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])

        # Build mapping/assignment matrix A (N_old x N_bins)
        A = np.zeros((N_old, N_bins))
        for i in range(N_bins):
            in_bin = (self.x_old >= self.bin_edges[i]) & (self.x_old < self.bin_edges[i+1])
            A[in_bin, i] = 1.0

        # Safety Check for Empty Bins
        points_per_bin = np.sum(A, axis=0)
        if np.any(points_per_bin == 0):
            empty_indices = np.where(points_per_bin == 0)[0]
            raise ValueError(f"Binning failed: Bins at indices {empty_indices} "
                             f"(centers: {x_new[empty_indices]}) contain no data points!")

        # Matrix compression via Cholesky decomposition solvers
        c, low = cho_factor(self.C_residual_old)
        invC_A = cho_solve((c, low), A)  # Equivalent to: C_inv @ A

        # Compute compressed residual covariance matrix (Fisher Information inversion)
        Fisher_matrix = A.T @ invC_A
        c_new, low_new = cho_factor(Fisher_matrix)
        self.C_residual_new = cho_solve((c_new, low_new), np.eye(N_bins))

        # Direct solver for the compressed residuals
        rhs = invC_A.T @ self.residual_old
        self.residual_new = self.C_residual_new @ rhs

        # Project parameters back out to physical space
        fid_f_new = fiducial(x_new)
        f_new = self.residual_new * fid_f_new

        # Scale binned matrix back to original physical units
        G_new = np.diag(fid_f_new)
        C_new = G_new @ self.C_residual_new @ G_new.T

        return f_new,C_new

    def get_derivatives(self,x_new,f_new,C_new):

        N_bins = len(x_new)

        D1 = np.zeros((N_bins, N_bins))
        D2 = np.zeros((N_bins, N_bins))
        J  = np.zeros((N_bins, N_bins))
        I  = np.eye(N_bins)

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

        # Evaluate physical secondary observables
        d1_new  = D1 @ f_new
        d2_new  = D2 @ f_new
        int_new = J @ f_new

        # =========================================================================
        # 3. JOINT COVARIANCE BLOCK MATRIX PROPAGATION
        # =========================================================================
        # Construct linear stack transformation mapping operator: T (4N_bins x N_bins)
        T = np.vstack([I, D1, D2, J])
        C_joint = T @ C_new @ T.T

        # Isolate the diagonal errors directly from block rows to maximize speed
        diag = np.sqrt(np.diag(C_joint))
        f_err   = diag[0          :   N_bins]
        d1_err  = diag[N_bins     : 2*N_bins]
        d2_err  = diag[2*N_bins   : 3*N_bins]
        int_err = diag[3*N_bins   : 4*N_bins]

        # Assemble requested DataFrame format
        data_df = pd.DataFrame({
            'x':       x_new,
            'f':       f_new,
            'd1':      d1_new,
            'd2':      d2_new,
            'int':     int_new,
            'f_err':   f_err,
            'd1_err':  d1_err,
            'd2_err':  d2_err,
            'int_err': int_err
        })

        # Generate unique structured block label coordinates
        prefixes = ['f', 'd1', 'd2', 'int']
        labels = [f'{p}_{i}' for p in prefixes for i in range(N_bins)]

        covmat_df = pd.DataFrame(C_joint, index=labels, columns=labels)

        return data_df,covmat_df



    def reconstruct(self,x_new,fiducial=None):
        """
        Performs flat residual binning followed by finite difference/integral 
        covariance propagation.
        
        x_new:    np.ndarray, arbitrary grid of coordinates for the new bins
        fiducial: callable, the reference physical model function
        """
        N_old  = len(self.x_old)
        N_bins = len(x_new)

        if fiducial == None:
            fiducial = lambda x: 1
        
        if self.method == 'FLAT':
            f_new,C_new = self.flat_residual_binning(x_new,fiducial)
        else:
            sys.exit('UNKNOWN BINNING METHOD: {}'.format(self.method))


        data_df,covmat_df = self.get_derivatives(self,x_new,f_new,C_new)

        return data_df, covmat_df
