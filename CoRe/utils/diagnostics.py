import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.linalg import block_diag

class Scorer:
    def __init__(self, df_reconstruction, df_joint_cov, eigen_trunc_factor=None, chatty=False):
        """
        Diagnostic scoring suite for CoRe reconstructions.
        
        df_reconstruction: pd.DataFrame containing the mean curves
        df_joint_cov:      pd.DataFrame containing the joint covariance matrix
        eigen_trunc_factor: float (F), if provided, truncates eigenmodes with eigenvalues 
                            smaller than max(eigenvalues) / F.
        chatty:             bool, toggles formatted console reporting
        """
        self.x = df_reconstruction['x'].values
        self.means = df_reconstruction
        self.df_joint_cov = df_joint_cov
        self.N = len(self.x)
        self.vars = [col for col in self.means.columns if col != 'x']
        self.eigen_trunc_factor = eigen_trunc_factor
        self.chatty = chatty

    # --- Internal Helpers ---
    def _get_indices(self, var_name, x_targets=None):
        if x_targets is None:
            return [f"{var_name}_{i}" for i in range(self.N)]
        pos = [np.argmin(np.abs(self.x - xi)) for xi in x_targets]
        return [f"{var_name}_{i}" for i in pos]

    def _print_formatted_report(self, title, results):
        """Prints a beautifully formatted, color-coded summary with orthogonal eigenmode breakdowns."""
        p = results.get('p_value', 0.0)
        
        if p < 0.01:
            color = "\033[91m"  # Red
            status = "POOR (Significant Tension)"
        elif p < 0.05:
            color = "\033[93m"  # Yellow
            status = "MARGINAL (High Tension)"
        else:
            color = "\033[92m"  # Green
            status = "GOOD (Consistent)"
        
        reset = "\033[0m"
        bold = "\033[1m"

        print(f"\n{bold}{'='*60}")
        print(f" SCORE REPORT: {title}")
        print(f"{'='*60}{reset}")
        print(f"Status (Full System): {color}{bold}{status}{reset}")
        print(f"Full Total Chi2:      {results.get('total_chi2', results.get('chi2', 0)):.2f}")
        print(f"Full Red.  Chi2:      {results.get('red_chi2', 0):.3f}")
        print(f"Full P-value:         {color}{p:.4f}{reset}")
        print(f"Full DOF:             {results.get('dof', 0)}")
        
        # Display Truncated Metrics if active
        if 'truncated_chi2' in results:
            p_trunc = results['truncated_p_value']
            if p_trunc < 0.01:      t_color = "\033[91m"
            elif p_trunc < 0.05:    t_color = "\033[93m"
            else:                   t_color = "\033[92m"
            
            print(f"-"*60)
            print(f"{bold}Truncated Subspace Metrics (Cutoff Factor F = {self.eigen_trunc_factor}):{reset}")
            print(f"Truncated Chi2:       {results['truncated_chi2']:.2f}")
            print(f"Truncated Red. Chi2:  {results['truncated_red_chi2']:.3f}")
            print(f"Truncated P-value:    {t_color}{p_trunc:.4f}{reset}")
            print(f"Truncated DOF:        {results['truncated_dof']} (Dropped {results['dof'] - results['truncated_dof']} noisy modes)")

        if 'breakdown' in results and title == "POINTWISE THEORY CONSISTENCY":
            print(f"\n{bold}Pointwise Component Breakdown:{reset}")
            for var, stats in results['breakdown'].items():
                print(f"  - {var: <5}: {stats['chi2_contrib']:8.2f} ({stats['percentage']:5.1f}%)")

        if 'eigenvalues' in results and 'chi2_per_mode' in results:
            evals = np.array(results['eigenvalues'])
            e_chi2 = np.array(results['chi2_per_mode'])
            total_chi2 = np.sum(e_chi2)
            
            print(f"\n{bold}Eigenmode Spectrum Breakdown (Sorted by Eigenvalue / Variance Descending):{reset}")
            print(f"{'Mode ID':<9} | {'Eigenvalue (Var)':<16} | {'Chi2 Contrib':<13} | {'Percentage':<10} | {'Status':<8}")
            print("-" * 75)
            
            max_display = 12
            max_val = evals[0] if len(evals) > 0 else 1.0
            
            for idx in range(min(len(evals), max_display)):
                pct = (e_chi2[idx] / total_chi2 * 100) if total_chi2 > 0 else 0.0
                
                # Flag if the mode was kept or dropped under truncation rules
                if self.eigen_trunc_factor is not None:
                    kept = evals[idx] >= (max_val * self.eigen_trunc_factor)
                    status_str = "KEPT" if kept else "DROPPED"
                    s_color = "\033[92m" if kept else "\033[90m"
                else:
                    status_str = "ACTIVE"
                    s_color = reset
                    
                print(f"Mode {idx:<4} | {evals[idx]:<16.2e} | {e_chi2[idx]:<13.2f} | {pct:5.1f}%     | {s_color}{status_str}{reset}")
                
            if len(evals) > max_display:
                remaining_chi2 = np.sum(e_chi2[max_display:])
                remaining_pct = (remaining_chi2 / total_chi2 * 100) if total_chi2 > 0 else 0.0
                print(f"Tail ({len(evals)-max_display} modes) | {'-':<16} | {remaining_chi2:<13.2f} | {remaining_pct:5.1f}%     | -")
                
        print(f"{bold}{'='*60}{reset}\n")

    # ---- Eigenmode Decomposer ---
    def break_eigenmodes(self, C_tot, diffvec):
        """
        Decomposes correlated residuals into mutually independent principal components,
        sorting strictly by Eigenvalue scale (variance size descending).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(C_tot)
        diffvec_projected = eigenvectors.T @ diffvec

        chi2_per_mode = (diffvec_projected**2) / eigenvalues
        
        # Sort indices by eigenvalue magnitude descending (highest variance to lowest)
        idx           = np.argsort(eigenvalues)[::-1]
        eigenvalues   = eigenvalues[idx]
        chi2_per_mode = chi2_per_mode[idx]

        return eigenvalues, chi2_per_mode

    def _apply_truncation(self, eigenvalues, chi2_per_mode, res):
        """Helper method to isolate truncated subspace contributions if active."""
        if self.eigen_trunc_factor is not None and len(eigenvalues) > 0:
            max_ev = eigenvalues[0]
            cutoff = max_ev * self.eigen_trunc_factor
            
            keep_mask = eigenvalues >= cutoff
            trunc_chi2 = float(np.sum(chi2_per_mode[keep_mask]))
            trunc_dof = int(np.sum(keep_mask))
            
            # Avoid division by zero if all modes are somehow discarded
            if trunc_dof > 0:
                trunc_red = trunc_chi2 / trunc_dof
                trunc_p = 1.0 - chi2.cdf(trunc_chi2, trunc_dof)
            else:
                trunc_red = 0.0
                trunc_p = 1.0
                
            res["truncated_chi2"] = trunc_chi2
            res["truncated_dof"] = trunc_dof
            res["truncated_red_chi2"] = trunc_red
            res["truncated_p_value"] = trunc_p
        return res

    # --- Scoring Methods ---
    def score_against_theory(self, theory_df):
        """Evaluates total joint consistency against a smooth analytical model utilizing full covariance information."""
        active_vars = [v for v in self.vars if v in theory_df.columns if v != 'x']
        all_r, all_labels = [], []
        
        for var in active_vars:
            all_r.append(self.means[var].values - theory_df[var].values)
            all_labels.extend(self._get_indices(var))
            
        R_total = np.concatenate(all_r)
        cov_sub = self.df_joint_cov.loc[all_labels, all_labels].values + np.eye(len(all_labels)) * 1e-11
        inv_cov = np.linalg.pinv(cov_sub)
        chi2_total = R_total.T @ inv_cov @ R_total

        res = {
            "total_chi2": float(chi2_total),
            "red_chi2": float(chi2_total / len(R_total)),
            "p_value": 1 - chi2.cdf(chi2_total, len(R_total)),
            "dof": len(R_total)
        }

        # Calculate exact orthogonal decomposition matrices
        res['eigenvalues'], res['chi2_per_mode'] = self.break_eigenmodes(cov_sub, R_total)
        
        # Apply truncation logic if active
        res = self._apply_truncation(res['eigenvalues'], res['chi2_per_mode'], res)

        if self.chatty:
            self._print_formatted_report("THEORY CONSISTENCY", res)
        return res

    def score_pointwise(self, theory_df):
        """
        Calculates a Chi-square score assuming NO correlation between points.
        This provides a 'visual-matching' score in a formal statistical format.
        """
        active_vars = [v for v in self.vars if v in theory_df.columns if v != 'x']
        all_r_sq = [] 
        breakdown = {}
        total_dof = 0

        for var in active_vars:
            mu = self.means[var].values
            y_theory = theory_df[var].values
            labels = self._get_indices(var)

            variance = np.diag(self.df_joint_cov.loc[labels, labels].values)
            pulls_sq = ((mu - y_theory)**2) / (variance + 1e-12)

            comp_chi2 = np.sum(pulls_sq)
            all_r_sq.append(pulls_sq)

            breakdown[var] = {
                "chi2_contrib": float(comp_chi2),
            }
            total_dof += self.N

        chi2_total = np.sum([b["chi2_contrib"] for b in breakdown.values()])

        for var in active_vars:
            breakdown[var]["percentage"] = (breakdown[var]["chi2_contrib"] / chi2_total * 100) if chi2_total > 0 else 0

        res = {
            "total_chi2": float(chi2_total),
            "red_chi2": float(chi2_total / total_dof),
            "p_value": 1 - chi2.cdf(chi2_total, total_dof),
            "dof": total_dof,
            "breakdown": breakdown
        }

        if self.chatty:
            self._print_formatted_report("POINTWISE THEORY CONSISTENCY", res)

        return res

    def score_against_data(self, datasets):
        """Scores the joint reconstruction directly against secondary raw data inputs."""
        all_r, gp_labels, ext_cov_list = [], [], []
        
        for ds in datasets:
            var = ds['type']
            x_ext = ds['df']['x'].values
            labels = self._get_indices(var, x_ext)
            
            mu_gp = self.means.iloc[[int(l.split('_')[1]) for l in labels]][var].values
            all_r.append(ds['df'][var].values - mu_gp)
            gp_labels.extend(labels)
            ext_cov_list.append(ds['cov'].values)
            
        R = np.concatenate(all_r)
        S_total = self.df_joint_cov.loc[gp_labels, gp_labels].values + block_diag(*ext_cov_list)
        chi2_val = R.T @ np.linalg.solve(S_total + np.eye(len(R))*1e-11, R)
        
        res = {
            "total_chi2": float(chi2_val),
            "red_chi2": float(chi2_val / len(R)),
            "p_value": 1 - chi2.cdf(chi2_val, len(R)),
            "dof": len(R)
        }

        res['eigenvalues'], res['chi2_per_mode'] = self.break_eigenmodes(S_total, R)
        
        # Apply truncation logic if active
        res = self._apply_truncation(res['eigenvalues'], res['chi2_per_mode'], res)

        if self.chatty:
            self._print_formatted_report("EXTERNAL DATA VALIDATION", res)
        return res
