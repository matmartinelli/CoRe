import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.linalg import block_diag

class Scorer:
    def __init__(self, df_reconstruction, df_joint_cov, chatty=False):
        self.x = df_reconstruction['x'].values
        self.means = df_reconstruction
        self.df_joint_cov = df_joint_cov
        self.N = len(self.x)
        self.vars = [col for col in self.means.columns if col != 'x']
        self.chatty = chatty

    # --- Internal Helpers ---
    def _get_indices(self, var_name, x_targets=None):
        if x_targets is None:
            return [f"{var_name}_{i}" for i in range(self.N)]
        pos = [np.argmin(np.abs(self.x - xi)) for xi in x_targets]
        return [f"{var_name}_{i}" for i in pos]

    def _print_formatted_report(self, title, results):
        """Prints a color-coded summary of the score."""
        p = results.get('p_value', 0.0)
        
        # Determine Color based on p-value
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

        print(f"\n{bold}{'='*50}")
        print(f" SCORE REPORT: {title}")
        print(f"{'='*50}{reset}")
        print(f"Status: {color}{bold}{status}{reset}")
        print(f"Total Chi2: {results.get('total_chi2', results.get('chi2', 0)):.2f}")
        print(f"Red.  Chi2: {results.get('red_chi2', 0):.3f}")
        print(f"P-value:    {color}{p:.4f}{reset}")
        print(f"DOF:        {results.get('dof', 0)}")
        
        if 'breakdown' in results:
            print(f"\n{bold}Residual Breakdown:{reset}")
            for var, stats in results['breakdown'].items():
                print(f"  - {var: <4}: {stats['chi2_contrib']:8.2f} ({stats['percentage']:5.1f}%)")
        print(f"{bold}{'='*50}{reset}\n")

    # --- Scoring Methods ---
    def score_against_theory(self, theory_df):
        active_vars = [v for v in self.vars if v in theory_df.columns if v != 'x']
        all_r, all_labels = [], []
        
        for var in active_vars:
            all_r.append(self.means[var].values - theory_df[var].values)
            all_labels.extend(self._get_indices(var))
            
        R_total = np.concatenate(all_r)
        cov_sub = self.df_joint_cov.loc[all_labels, all_labels].values + np.eye(len(all_labels)) * 1e-11
        inv_cov = np.linalg.pinv(cov_sub)
        chi2_total = R_total.T @ inv_cov @ R_total
        
        breakdown = {}
        for i, var in enumerate(active_vars):
            r_part = all_r[i]
            inv_block = inv_cov[i*self.N : (i+1)*self.N, i*self.N : (i+1)*self.N]
            comp_chi2 = r_part.T @ inv_block @ r_part
            breakdown[var] = {"chi2_contrib": float(comp_chi2), "percentage": float(comp_chi2/chi2_total*100)}

        res = {
            "total_chi2": float(chi2_total),
            "red_chi2": float(chi2_total / len(R_total)),
            "p_value": 1 - chi2.cdf(chi2_total, len(R_total)),
            "dof": len(R_total),
            #"breakdown": breakdown #MMmod: fix breakdown!
        }

        if self.chatty:
            self._print_formatted_report("THEORY CONSISTENCY", res)
        return res

    def score_pointwise(self, theory_df):
        """
        Calculates a Chi-square score assuming NO correlation between points.
        This provides a 'visual-matching' score in a formal statistical format.
        """
        active_vars = [v for v in self.vars if v in theory_df.columns if v != 'x']
        all_r_sq = [] # Stores (r/sigma)^2
        breakdown = {}

        total_dof = 0

        for var in active_vars:
            mu = self.means[var].values
            y_theory = theory_df[var].values
            labels = self._get_indices(var)

            # Extract diagonal only (the pointwise variance)
            variance = np.diag(self.df_joint_cov.loc[labels, labels].values)

            # (Mean - Theory)^2 / Variance
            pulls_sq = ((mu - y_theory)**2) / (variance + 1e-12)

            comp_chi2 = np.sum(pulls_sq)
            all_r_sq.append(pulls_sq)

            # Save breakdown for this component
            breakdown[var] = {
                "chi2_contrib": float(comp_chi2),
                # Percentage will be calculated after total_chi2 is known
            }
            total_dof += self.N

        # Calculate Global Totals
        chi2_total = np.sum([b["chi2_contrib"] for b in breakdown.values()])

        # Fill in percentages now that we have the total
        for var in active_vars:
            breakdown[var]["percentage"] = (breakdown[var]["chi2_contrib"] / chi2_total * 100) if chi2_total > 0 else 0

        res = {
            "total_chi2": float(chi2_total),
            "red_chi2": float(chi2_total / total_dof),
            "p_value": 1 - chi2.cdf(chi2_total, total_dof),
            "dof": total_dof,
            #"breakdown": breakdown #MMmod: TODO fix breakdown!
        }

        if self.chatty:
            self._print_formatted_report("POINTWISE THEORY CONSISTENCY", res)

        return res

    def score_against_data(self, datasets):
        all_r, gp_labels, ext_cov_list = [], [], []
        
        for ds in datasets:
            var = ds['type']
            x_ext = ds['df']['x'].values
            labels = self._get_indices(var, x_ext)
            # Map indices correctly
            mu_gp = self.means.iloc[[int(l.split('_')[1]) for l in labels]][var].values
            all_r.append(ds['df'][var].values - mu_gp)
            gp_labels.extend(labels)
            ext_cov_list.append(ds['cov'].values)
            
        R = np.concatenate(all_r)
        S_total = self.df_joint_cov.loc[gp_labels, gp_labels].values + block_diag(*ext_cov_list)
        chi2_val = R.T @ np.linalg.solve(S_total + np.eye(len(R))*1e-11, R)
        
        res = {
            "chi2": float(chi2_val),
            "red_chi2": float(chi2_val / len(R)),
            "p_value": 1 - chi2.cdf(chi2_val, len(R)),
            "dof": len(R)
        }
        if self.chatty:
            self._print_formatted_report("EXTERNAL DATA VALIDATION", res)
        return res
