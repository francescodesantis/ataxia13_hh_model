import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/Users/francescodesantis/Library/CloudStorage/OneDrive-PolitecnicodiMilano/work/sound_localization/ATAXIA_PROJECT/Fransescos Variables/"

GROUPS = {
    "MNTB_direct":   os.path.join(BASE_DIR, "MNTB", "direct"),
    "MNTB_indirect": os.path.join(BASE_DIR, "MNTB", "indirect"),
    "LSO_direct":    os.path.join(BASE_DIR, "LSO",  "direct"),
    "LSO_indirect":  os.path.join(BASE_DIR, "LSO",  "indirect"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA                  = 0.05
EFFECT_SIZE_THRESHOLD  = 0.8
IQR_MULTIPLIER         = 1.5   # standard Tukey fence; raise to 3.0 for "extreme" outliers only

# ------------------------------------------------------------
# Final output column order (as requested)
# ------------------------------------------------------------
FINAL_COLUMN_ORDER = [
    "parameter",
    "n_WT",
    "n_HOM",
    "test",
    "WT_normal",
    "HOM_normal",
    "raw_p_value",
    "BH_corrected_p",
    "effect_size_Cohens_d",
    "decision",
    "WT_mean",
    "HOM_mean",
    "Common_mean",
    "WT_median",
    "HOM_median",
    "Common_median",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_excel_file(filepath):
    """Loads an Excel file with columns: WT | Hom."""
    df = pd.read_excel(filepath)
    if len(df.columns) < 2:
        raise ValueError(f"File {filepath} does not contain at least 2 columns")
    wt  = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
    hom = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().values
    return wt, hom


# ------------------------------------------------------------
# Outlier detection
# ------------------------------------------------------------

def detect_outliers(data, multiplier=IQR_MULTIPLIER):
    """
    Identifies outliers in `data` using the Tukey IQR fence.

    Returns
    -------
    clean   : np.ndarray  — values inside [lower, upper]
    mask    : np.ndarray  — boolean mask of kept values (same length as data)
    lower   : float       — lower fence  (Q1 - multiplier * IQR)
    upper   : float       — upper fence  (Q3 + multiplier * IQR)
    n_removed : int       — number of outliers removed
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr     = q3 - q1
    lower   = q1 - multiplier * iqr
    upper   = q3 + multiplier * iqr
    mask    = (data >= lower) & (data <= upper)
    clean   = data[mask]
    return clean, mask, lower, upper, int((~mask).sum())


def remove_outliers_per_group(wt, hom, multiplier=IQR_MULTIPLIER):
    """
    Applies outlier detection independently to WT and HOM.

    Returns a dict with cleaned arrays and outlier diagnostics.
    """
    wt_clean,  wt_mask,  wt_lo,  wt_hi,  wt_n  = detect_outliers(wt,  multiplier)
    hom_clean, hom_mask, hom_lo, hom_hi, hom_n = detect_outliers(hom, multiplier)

    outlier_info = {
        "WT_outliers_removed":  wt_n,
        "HOM_outliers_removed": hom_n,
        "WT_fence_lower":       round(wt_lo,  4),
        "WT_fence_upper":       round(wt_hi,  4),
        "HOM_fence_lower":      round(hom_lo, 4),
        "HOM_fence_upper":      round(hom_hi, 4),
        "WT_outlier_values":    list(wt[~wt_mask]),
        "HOM_outlier_values":   list(hom[~hom_mask]),
    }

    if wt_n > 0:
        print(f"  [outliers] WT:  removed {wt_n} value(s): {outlier_info['WT_outlier_values']}")
    if hom_n > 0:
        print(f"  [outliers] HOM: removed {hom_n} value(s): {outlier_info['HOM_outlier_values']}")

    return wt_clean, hom_clean, outlier_info


# ------------------------------------------------------------
# Cohen's d
# ------------------------------------------------------------

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std


# ------------------------------------------------------------
# Normality test
# ------------------------------------------------------------

def is_normal(data):
    if len(data) < 3:
        return False
    stat, p = stats.shapiro(data)
    return p > 0.05


# ------------------------------------------------------------
# Statistical test selection
# ------------------------------------------------------------

def compare_groups(wt, hom):
    wt_normal  = is_normal(wt)
    hom_normal = is_normal(hom)

    if wt_normal and hom_normal:
        test_name = "Welch_t_test"
        stat, p_value = stats.ttest_ind(wt, hom, equal_var=False, nan_policy="omit")
    else:
        test_name = "Mann_Whitney_U"
        stat, p_value = stats.mannwhitneyu(wt, hom, alternative="two-sided")

    return {
        "test":       test_name,
        "p_value":    p_value,
        "effect_size": abs(cohens_d(wt, hom)),
        "wt_normal":  wt_normal,
        "hom_normal": hom_normal,
    }


# ------------------------------------------------------------
# Plot distributions
# ------------------------------------------------------------

def plot_distribution(parameter_name, wt, hom, output_folder):
    fig, ax = plt.subplots(figsize=(6, 6))
    data   = [wt, hom]
    parts  = ax.violinplot(data, showmeans=False, showmedians=True)
    rng    = np.random.default_rng(42)
    for i, group in enumerate(data, start=1):
        jitter = rng.normal(0, 0.04, size=len(group))
        ax.scatter(np.full(len(group), i) + jitter, group, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["WT", "HOM"])
    ax.set_title(parameter_name)
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{parameter_name}.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# Representative values
# ------------------------------------------------------------

def representative_values(wt, hom, separate):
    if separate:
        return {
            "WT_mean":     np.mean(wt),   "HOM_mean":    np.mean(hom),  "Common_mean":   np.nan,
            "WT_median":   np.median(wt), "HOM_median":  np.median(hom), "Common_median": np.nan,
        }
    combined = np.concatenate([wt, hom])
    return {
        "WT_mean":     np.nan, "HOM_mean":    np.nan, "Common_mean":   np.mean(combined),
        "WT_median":   np.nan, "HOM_median":  np.nan, "Common_median": np.median(combined),
    }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_folder(folder_name, folder_path):
    print(f"\n================================================")
    print(f"Analyzing: {folder_name}")
    print(f"================================================")

    excel_files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    if not excel_files:
        print("No Excel files found")
        return

    plots_dir = os.path.join(OUTPUT_DIR, folder_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    raw_pvalues        = []
    temporary_results  = []

    # --------------------------------------------------------
    # FIRST PASS — raw p-values
    # --------------------------------------------------------
    for filepath in excel_files:
        parameter_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            wt_raw, hom_raw = load_excel_file(filepath)

            # ---- outlier removal ----------------------
            wt, hom, outlier_info = remove_outliers_per_group(wt_raw, hom_raw)

            # skip parameter if too few observations remain
            if len(wt) < 3 or len(hom) < 3:
                print(f"  [skip] {parameter_name}: too few observations after outlier removal "
                      f"(WT={len(wt)}, HOM={len(hom)})")
                continue

            plot_distribution(parameter_name, wt, hom, plots_dir)

            stats_result = compare_groups(wt, hom)
            raw_pvalues.append(stats_result["p_value"])

            temporary_results.append({
                "parameter":    parameter_name,
                "wt":           wt,
                "hom":          hom,
                "wt_raw_n":     len(wt_raw),
                "hom_raw_n":    len(hom_raw),
                "outlier_info": outlier_info,
                "stats":        stats_result,
            })

        except Exception as e:
            print(f"  [error] {parameter_name}: {e}")

    if not raw_pvalues:
        print("No valid parameters to analyse.")
        return

    # --------------------------------------------------------
    # BH FDR correction
    # --------------------------------------------------------
    reject, corrected_pvalues, _, _ = multipletests(raw_pvalues, alpha=ALPHA, method="fdr_bh")

    # --------------------------------------------------------
    # SECOND PASS — build final table
    # --------------------------------------------------------
    results = []
    for temp, adj_p, rej in zip(temporary_results, corrected_pvalues, reject):
        wt           = temp["wt"]
        hom          = temp["hom"]
        stats_result = temp["stats"]
        effect_size  = stats_result["effect_size"]

        separate = (adj_p < ALPHA) and (effect_size > EFFECT_SIZE_THRESHOLD)
        decision = "SEPARATE" if separate else "POOL"
        reps     = representative_values(wt, hom, separate)

        results.append({
            "parameter":             temp["parameter"],
            "n_WT":                  len(wt),
            "n_HOM":                 len(hom),
            "test":                  stats_result["test"],
            "WT_normal":             stats_result["wt_normal"],
            "HOM_normal":            stats_result["hom_normal"],
            "raw_p_value":           stats_result["p_value"],
            "BH_corrected_p":        adj_p,
            "effect_size_Cohens_d":  effect_size,
            "decision":              decision,
            "WT_mean":               reps["WT_mean"],
            "HOM_mean":              reps["HOM_mean"],
            "Common_mean":           reps["Common_mean"],
            "WT_median":             reps["WT_median"],
            "HOM_median":            reps["HOM_median"],
            "Common_median":         reps["Common_median"],
        })

    # --------------------------------------------------------
    # BUILD DATAFRAME, APPLY ROW ORDER, APPLY COLUMN ORDER
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df[FINAL_COLUMN_ORDER]

    # --------------------------------------------------------
    # SAVE TABLE
    # --------------------------------------------------------
    output_excel = os.path.join(OUTPUT_DIR, f"{folder_name}_statistics_summary.xlsx")
    results_df.to_excel(output_excel, index=False)
    print(f"\nSaved summary table: {output_excel}")
    print(f"Saved plots:         {plots_dir}")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    for folder_name, folder_path in GROUPS.items():
        analyze_folder(folder_name, folder_path)
    print("\n================================================")
    print("ALL ANALYSES COMPLETED")
    print("================================================")